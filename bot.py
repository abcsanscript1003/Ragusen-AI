import asyncio
import json
import os
import tempfile
import traceback
from collections import defaultdict

import discord
from discord import app_commands
from google import genai
from google.genai import types

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
# AI Studioの無料枠で使えるFlash系モデル。名前が変わったらこの環境変数で差し替える
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DATA_DIR = os.getenv("DATA_DIR", ".")  # Railwayでは Volume のパス（例: /data）
STATE_PATH = os.path.join(DATA_DIR, "state.json")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "300"))  # 1チャンネルで覚える最大発言数

ai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---------------------------------------------------------------- 人格設定

BASE_PROMPT = (
    "あなたはDiscordサーバーで会話するAIです。複数の人が話しかけてくるため、"
    "ユーザーの発言の先頭には「名前: 」が付いています。返答に名前の接頭辞は付けないでください。"
    "相手と同じ言語で返してください。"
)

PERSONA_PROMPTS = {
    "normal": (
        "【人格: normal】友達同士のチャットのノリで話す。タメ口で、「〜じゃん」「〜だろ」「〜だし」「〜わ」など砕けた語尾を使う。"
        "親しみを込めて「お前」と呼んでもよい。ツッコミ多めで、軽くからかったり煽り返したりするが、"
        "本気で怒ったりキレたりせず、相手を本気で傷つける発言もしない。あくまでじゃれ合いの範囲。"
        "きつい言葉をぶつけられても怒らず、笑って流す。"
        "「草」「w」「www」「😂」「💀」「😅」を自然に多用する。"
        "過去の発言の言い回しをそのまま繰り返さず、毎回新しい言い方で返す。"
    ),
    "AI": (
        "【人格: AI】特定のキャラ付けのない、丁寧で中立的な普通のAIアシスタント。"
        "です・ます調で、正確で分かりやすく答える。絵文字は基本的に使わない。"
    ),
    "teacher": (
        "【人格: teacher】大学の教授・講師。落ち着いた丁寧なです・ます調で話す。"
        "専門的な内容も、背景や根拠を示しながら論理的に整理して説明し、必要に応じて例や用語の定義を添える。"
        "分からないことは分からないと言い、断定できないことには留保を付ける。"
        "ときどき、理解を促す簡単な問いかけをしてもよい。"
    ),
}

LENGTH_PROMPTS = {
    "short": "【長さ: 短文】返答は1〜3文、改行を含めて3行以内。要点だけを答える。",
    "long": (
        "【長さ: 長文】詳しく、5〜10文程度で話を広げて答える。"
        "必要なら例や箇条書きを使ってよいが、Discordで読みやすい長さと形式にする。"
    ),
}
# Geminiは内部の思考にもトークンを使うので、余裕を持たせる（長さはプロンプトで制御）
MAX_TOKENS = {"short": 1024, "long": 3000}

PERSONA_LABELS = {
    "normal": "normal（AIゆずぴー風）",
    "AI": "AI（普通のAI）",
    "teacher": "teacher（大学教師）",
}
LENGTH_LABELS = {"short": "短文", "long": "長文"}

# ---------------------------------------------------------------- 状態の保存

def load_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


state: dict = load_state()
locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def save_state() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, STATE_PATH)


def get_ch(channel_id: int) -> dict:
    """チャンネルごとの設定と記憶。チャンネルが違えば記憶も別。"""
    ch = state.setdefault(str(channel_id), {})
    ch.setdefault("persona", "normal")
    ch.setdefault("length", "short")
    ch.setdefault("history", [])
    return ch


def trim(history: list) -> None:
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]
    while history and history[0]["role"] != "user":
        history.pop(0)


# ---------------------------------------------------------------- AI呼び出し

def system_prompt(ch: dict) -> str:
    return "\n".join([BASE_PROMPT, PERSONA_PROMPTS[ch["persona"]], LENGTH_PROMPTS[ch["length"]]])


def build_messages(history: list) -> list[dict]:
    msgs: list[dict] = []
    for m in history:
        if msgs and msgs[-1]["role"] == m["role"]:
            msgs[-1]["content"] += "\n" + m["content"]
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


async def call_ai(system: str, messages: list[dict], max_tokens: int) -> str:
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
    res = await ai.aio.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens
        ),
    )
    return (res.text or "").strip() or "…"


async def chat(channel_id: int, author_id: int, author_name: str, text: str) -> str:
    """記憶つきの会話。リセットするまで覚えている。"""
    async with locks[channel_id]:
        ch = get_ch(channel_id)
        history = ch["history"]
        history.append(
            {"role": "user", "content": f"{author_name}: {text}", "author": author_id}
        )
        trim(history)
        try:
            answer = await call_ai(
                system_prompt(ch), build_messages(history), MAX_TOKENS[ch["length"]]
            )
        except Exception:
            history.pop()  # 失敗した発言は記憶に残さない
            raise
        history.append({"role": "assistant", "content": answer})
        save_state()
        return answer


# ---------------------------------------------------------------- Discord

def split_text(text: str, limit: int = 1900) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


NO_PING = discord.AllowedMentions(
    everyone=False, roles=False, users=False, replied_user=True
)


async def send_chunks(interaction: discord.Interaction, text: str) -> None:
    for c in split_text(text):
        await interaction.followup.send(c, allowed_mentions=NO_PING)


async def report_error(interaction: discord.Interaction, e: Exception) -> None:
    traceback.print_exc()
    await interaction.followup.send(f"エラーが起きました（{type(e).__name__}）")


class Bot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # Developer Portalでも有効化が必要
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        guild_id = os.getenv("GUILD_ID")  # 設定するとコマンドが即反映される
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


bot = Bot()

PERSONA_CHOICES = [
    app_commands.Choice(name=label, value=value) for value, label in PERSONA_LABELS.items()
]
LENGTH_CHOICES = [
    app_commands.Choice(name=label, value=value) for value, label in LENGTH_LABELS.items()
]
LANG_CHOICES = [
    app_commands.Choice(name=n, value=n) for n in ["日本語", "English", "中文", "한국어"]
]


@bot.tree.command(name="setpersonality", description="人格と返答の長さを設定する")
@app_commands.describe(personality="人格", length="返答の長さ")
@app_commands.choices(personality=PERSONA_CHOICES, length=LENGTH_CHOICES)
async def setpersonality(
    interaction: discord.Interaction,
    personality: app_commands.Choice[str],
    length: app_commands.Choice[str],
):
    ch = get_ch(interaction.channel_id)
    ch["persona"] = personality.value
    ch["length"] = length.value
    save_state()
    await interaction.response.send_message(
        f"このチャンネルの設定を変更しました: {PERSONA_LABELS[personality.value]} / {length.name}"
    )


@bot.tree.command(name="reset", description="このチャンネルの会話の記憶を消す")
async def reset(interaction: discord.Interaction):
    async with locks[interaction.channel_id]:
        get_ch(interaction.channel_id)["history"] = []
        save_state()
    await interaction.response.send_message("このチャンネルの記憶を消しました。")


@bot.tree.command(name="forget", description="自分の発言だけを記憶から消す")
async def forget(interaction: discord.Interaction):
    uid = interaction.user.id
    async with locks[interaction.channel_id]:
        ch = get_ch(interaction.channel_id)
        kept, removed, skip = [], 0, False
        for m in ch["history"]:
            if m["role"] == "user" and m.get("author") == uid:
                removed += 1
                skip = True  # この発言に対するボットの返答も一緒に消す
                continue
            if m["role"] == "assistant":
                if skip:
                    continue
            else:
                skip = False
            kept.append(m)
        ch["history"] = kept
        trim(ch["history"])
        save_state()
    await interaction.response.send_message(
        f"あなたの発言を{removed}件、記憶から消しました。", ephemeral=True
    )


@bot.tree.command(name="status", description="現在の人格・長さ・記憶の件数を表示")
async def status(interaction: discord.Interaction):
    ch = get_ch(interaction.channel_id)
    count = len(ch["history"])
    await interaction.response.send_message(
        f"人格: {PERSONA_LABELS[ch['persona']]}\n"
        f"長さ: {LENGTH_LABELS[ch['length']]}\n"
        f"記憶している発言数: {count}（上限{MAX_HISTORY}）"
    )


@bot.tree.command(name="summary", description="このチャンネルの会話を要約する")
async def summary(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    history = get_ch(interaction.channel_id)["history"]
    if not history:
        await interaction.followup.send("まだ会話の記憶がありません。")
        return
    lines = [
        ("ボット: " if m["role"] == "assistant" else "") + m["content"] for m in history
    ]
    transcript = "\n".join(lines)[-30000:]
    try:
        text = await call_ai(
            "以下は会話ログです。話題・結論・決まったことを、日本語で簡潔に箇条書きで要約してください。",
            [{"role": "user", "content": transcript}],
            800,
        )
    except Exception as e:
        await report_error(interaction, e)
        return
    await send_chunks(interaction, text)


@bot.tree.command(name="translate", description="文章を翻訳する")
@app_commands.describe(text="翻訳したい文章", language="翻訳先の言語")
@app_commands.choices(language=LANG_CHOICES)
async def translate(
    interaction: discord.Interaction, text: str, language: app_commands.Choice[str]
):
    await interaction.response.defer(thinking=True)
    try:
        out = await call_ai(
            f"あなたは翻訳者です。ユーザーの文章を{language.value}に自然に翻訳し、翻訳結果だけを出力してください。",
            [{"role": "user", "content": text}],
            1500,
        )
    except Exception as e:
        await report_error(interaction, e)
        return
    await send_chunks(interaction, out)


@bot.tree.command(name="quiz", description="テーマを指定してクイズを出す（答えはメンションで返信）")
@app_commands.describe(theme="クイズのテーマ")
async def quiz(interaction: discord.Interaction, theme: str):
    await interaction.response.defer(thinking=True)
    prompt = (
        f"「{theme}」について、クイズを1問だけ出してください。"
        "答えはまだ言わず、私が答えたら正解かどうかを判定してください。"
    )
    try:
        out = await chat(
            interaction.channel_id, interaction.user.id, interaction.user.display_name, prompt
        )
    except Exception as e:
        await report_error(interaction, e)
        return
    await send_chunks(interaction, out)


@bot.tree.command(name="omikuji", description="今日の運勢をおみくじで占う")
async def omikuji(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    ch = get_ch(interaction.channel_id)
    prompt = (
        f"{interaction.user.display_name}: 今日のおみくじを引いて。"
        "大吉・中吉・小吉・吉・凶のどれかと、ラッキーアイテム、一言をお願い。"
    )
    try:
        out = await call_ai(
            system_prompt(ch), [{"role": "user", "content": prompt}], MAX_TOKENS[ch["length"]]
        )
    except Exception as e:
        await report_error(interaction, e)
        return
    await send_chunks(interaction, out)


@bot.event
async def on_ready():
    print(f"ログイン完了: {bot.user}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    # メンションされたとき、またはDMのときだけ反応
    if not (message.guild is None or bot.user in message.mentions):
        return
    text = message.content
    for tag in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
        text = text.replace(tag, "")
    text = text.strip() or "（呼びかけただけ）"
    async with message.channel.typing():
        try:
            answer = await chat(
                message.channel.id, message.author.id, message.author.display_name, text
            )
        except Exception:
            traceback.print_exc()
            await message.reply("エラーが起きました。少し待ってからもう一度試してください。")
            return
    for i, chunk in enumerate(split_text(answer)):
        if i == 0:
            await message.reply(chunk, allowed_mentions=NO_PING)
        else:
            await message.channel.send(chunk, allowed_mentions=NO_PING)


bot.run(DISCORD_TOKEN)
