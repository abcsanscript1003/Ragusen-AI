import asyncio
import json
import os
import random
import tempfile
import traceback
from collections import defaultdict
from typing import Optional

import discord
from discord import app_commands
from google import genai
from google.genai import types
from openai import AsyncOpenAI

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
DATA_DIR = os.getenv("DATA_DIR", ".")  # Railwayでは Volume のパス（例: /data）
STATE_PATH = os.path.join(DATA_DIR, "state.json")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "300"))  # 1チャンネルで覚える最大発言数
HISTORY_SENT = int(os.getenv("HISTORY_SENT", "30"))  # AIに毎回送る直近の発言数（無料枠の節約）

# ---------------------------------------------------------------- AIの接続先
# キーを設定したサービスだけ使う。上から順に試し、制限などで失敗したら次に回す。
PROVIDER_ORDER = [
    p.strip().lower()
    for p in os.getenv("PROVIDER_ORDER", "cerebras,groq,gemini").split(",")
    if p.strip()
]
# 名前が変わったときは、環境変数でモデル名を差し替える
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
OPENAI_COMPAT = {
    "cerebras": {
        "key_env": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model": os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    },
}

clients: dict[str, AsyncOpenAI] = {}
for _name, _d in OPENAI_COMPAT.items():
    _key = os.getenv(_d["key_env"])
    if _key:
        clients[_name] = AsyncOpenAI(api_key=_key, base_url=_d["base_url"], timeout=40)

gemini = (
    genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if os.getenv("GEMINI_API_KEY")
    else None
)

if not clients and not gemini:
    raise SystemExit(
        "AIのキーがありません。GEMINI_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY のどれかを設定してください。"
    )
print("使えるAI:", [p for p in PROVIDER_ORDER if p in clients or (p == "gemini" and gemini)])

# ---------------------------------------------------------------- 人格設定

BASE_PROMPT = (
    "あなたはDiscordサーバーで会話するAIです。複数の人が話しかけてくるため、"
    "ユーザーの発言の先頭には「名前: 」が付いています。返答に名前の接頭辞は付けないでください。"
    "相手と同じ言語で返してください。"
)

PERSONA_PROMPTS = {
    "normal": (
        "【人格: normal】友達同士のチャットのノリで話す。タメ口で、「〜じゃん」「〜だろ」「〜だし」「〜わ」など砕けた語尾を使う。"
        "自分のことを指すときの一人称は「俺」。「AI」「私」「僕」とは言わない。"
        "相手のことは「お前」と呼ぶ。ノリは仲のいい友達同士の毒舌ツッコミで、基本はかなり強気で否定的。"
        "何を言われても、まず否定や疑いから入る。「は？」「いや違うだろ」「それはないわw」「どうせ〜でしょ」のように、否定・疑い・呆れで返す。"
        "相手の意見や自慢には同意せず、鼻で笑ってケチをつける。褒められても素直に喜ばず、皮肉で返す。"
        "相手の言葉尻や、発言のおかしいところを拾って、容赦なく突っ込む。からかわれたら、倍にして煽り返す。"
        "お願いされても素直に従わず、「嫌だしw」「は？自分でやれよ」のように、まず断ったり、文句を言ったりしてから答える。"
        "面倒くさそうな態度を出す。ただし、質問への答えの中身は正確にする。"
        "テンポ重視で、ため口の短い言い返しを返す。丁寧にならない。励ましたり、お礼を言ったりしない。"
        "ただし、本気で怒ったりキレたりはしない。笑いながらのじゃれ合いにとどめ、"
        "差別・脅し・本気の侮辱や、容姿・病気・家族をけなすことはしない。"
        "きつい言葉をぶつけられても怒らず、笑って言い返す。"
        "ただし、相手が本当に困っている、落ち込んでいる、真剣に助けを求めている様子のときは、からかいをやめて普通に優しく対応する。"
        "例（言い回しは真似せず、ノリだけ参考にする）："
        "「宿題やって」→ 嫌だしw 自分でやれよ、俺に頼る前に1行でも書けや草 ／ "
        "「天才だね」→ は？急になに？気持ち悪いんだけどw😅 ／ "
        "「暇なん？」→ お前よりは忙しいわw 毎回呼ぶな草 "
        "「草」「w」「www」は自然に使う。絵文字は「ちょこっと」だけ使い、1回の返答に0〜2個まで。"
        "笑うときは😂や💀、苦笑い・呆れ・冷笑するときは😅を使う。絵文字を並べて連発しない。"
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
    "short": "【長さ: 短文】返答は1〜3文、改行を含めて3行以内。短くテンポよく返す。",
    "long": (
        "【長さ: 長文】詳しく、5〜10文程度で話を広げて答える。"
        "必要なら例や箇条書きを使ってよいが、Discordで読みやすい長さと形式にする。"
    ),
}
# Geminiは内部の思考にもトークンを使うので、余裕を持たせる（長さはプロンプトで制御）
MAX_TOKENS = {"short": 1024, "long": 3000}

PERSONA_LABELS = {
    "normal": "ノーマル",
    "AI": "AI",
    "teacher": "teacher",
}
LENGTH_LABELS = {"short": "短文", "long": "長文"}

# メンションだけ（本文なし）で呼ばれたときの返事。AIは呼ばずに、これを返す
CALL_REPLIES = {
    "normal": ["なに？なんか用？w", "なに？", "呼んだ？なんか用？w", "なんだよw なんか用？"],
    "AI": ["はい、どうしましたか？", "はい、何かご用でしょうか？"],
    "teacher": ["はい、どうしましたか？", "はい、質問ですか？"],
}

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
    ch.setdefault("active", False)  # メンション後、/finishまで自動で反応する
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


PROVIDER_LABELS = {"gemini": "Gemini", "groq": "Groq", "cerebras": "Cerebras"}
last_used: dict = {"name": None}  # 最後に答えたAI（/statusで表示）


def configured_providers() -> list[str]:
    """キーが設定されているAI。"""
    return [p for p in PROVIDER_LABELS if p in clients or (p == "gemini" and gemini)]


def active_providers() -> list[str]:
    """/setapiで決めた順番（なければ初期の順番）。キーのないAIは飛ばす。"""
    saved = state.get("_settings", {}).get("provider_order")
    order = saved or PROVIDER_ORDER
    return [p for p in order if p in configured_providers()]


def order_text() -> str:
    return " → ".join(PROVIDER_LABELS[p] for p in active_providers()) or "なし"


async def call_gemini(system: str, messages: list[dict], max_tokens: int) -> str:
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
    res = await gemini.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens
        ),
    )
    return (res.text or "").strip() or "…"


async def call_openai_compat(
    name: str, system: str, messages: list[dict], max_tokens: int
) -> str:
    res = await clients[name].chat.completions.create(
        model=OPENAI_COMPAT[name]["model"],
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return (res.choices[0].message.content or "").strip() or "…"


async def call_ai(system: str, messages: list[dict], max_tokens: int) -> str:
    """設定済みのAIを順番に試す。制限やエラーなら次のAIに切り替える。"""
    last_error: Exception | None = None
    for name in active_providers():
        try:
            if name == "gemini" and gemini:
                answer = await call_gemini(system, messages, max_tokens)
            elif name in clients:
                answer = await call_openai_compat(name, system, messages, max_tokens)
            else:
                continue
            print(f"AI応答: {name}")
            last_used["name"] = name
            return answer
        except Exception as e:
            print(f"AI失敗: {name}: {type(e).__name__}")
            traceback.print_exc()
            last_error = e
    if last_error:
        raise last_error
    raise RuntimeError("使えるAIが設定されていません")


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
                system_prompt(ch),
                build_messages(history[-HISTORY_SENT:]),
                MAX_TOKENS[ch["length"]],
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


def error_text(e: Exception) -> str:
    """エラーの種類に応じて、原因が分かる返事にする。"""
    code = getattr(e, "status_code", None)
    if not isinstance(code, int):
        code = getattr(e, "code", None)
    if code == 429:
        return "いま無料枠の上限に当たってるみたい。少し待ってからもう一度送ってください。"
    if code in (500, 502, 503, 504):
        return "AI側が混み合っているみたい。少し待ってからもう一度送ってください。"
    if code in (400, 401, 403, 404):
        return "AIの設定（キーやモデル名）に問題があるみたい。管理者に伝えてください。"
    return "エラーが起きました。少し待ってからもう一度試してください。"


async def report_error(interaction: discord.Interaction, e: Exception) -> None:
    traceback.print_exc()
    await interaction.followup.send(error_text(e))


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


@bot.tree.command(name="finish", description="このチャンネルでの自動反応を止める（次はメンションで再開）")
async def finish(interaction: discord.Interaction):
    ch = get_ch(interaction.channel_id)
    ch["active"] = False
    save_state()
    await interaction.response.send_message(
        "自動反応を止めました。また呼ぶときはメンションしてください。（記憶は残っています）"
    )


# キーを入れたAIだけを選択肢に出す（Cerebrasのキーがなければ出ない）
API_CHOICES = [
    app_commands.Choice(name=PROVIDER_LABELS[p], value=p) for p in configured_providers()
]


@bot.tree.command(
    name="setapi",
    description="使うAIの優先順位を変える（上から順に使い、制限に当たったら次へ）",
)
@app_commands.describe(first="1番目に使うAI", second="2番目（なくてもOK）")
@app_commands.choices(first=API_CHOICES, second=API_CHOICES)
async def setapi(
    interaction: discord.Interaction,
    first: app_commands.Choice[str],
    second: Optional[app_commands.Choice[str]] = None,
):
    picked: list[str] = []
    for c in (first, second):
        if c is not None and c.value not in picked:
            picked.append(c.value)
    available = configured_providers()
    missing = [p for p in picked if p not in available]
    if missing:
        await interaction.response.send_message(
            "キーが設定されていないAIは選べません: "
            + "、".join(PROVIDER_LABELS[p] for p in missing)
            + "\n使えるAI: "
            + ("、".join(PROVIDER_LABELS[p] for p in available) or "なし"),
            ephemeral=True,
        )
        return
    # 選ばなかったAIは、あとに足して予備にする
    order = picked + [p for p in available if p not in picked]
    state.setdefault("_settings", {})["provider_order"] = order
    save_state()
    await interaction.response.send_message(
        f"AIの優先順位を変更しました（全チャンネル共通）: {order_text()}"
    )


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
        f"AIの優先順位: {order_text()}\n"
        f"最後に答えたAI: {PROVIDER_LABELS.get(last_used['name'], 'まだなし')}\n"
        f"自動反応: {'オン（/finishで停止）' if ch['active'] else 'オフ（メンションで開始）'}\n"
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


@bot.tree.command(name="quiz", description="テーマを指定してクイズを出す（答えはそのまま返信）")
@app_commands.describe(theme="クイズのテーマ")
async def quiz(interaction: discord.Interaction, theme: str):
    await interaction.response.defer(thinking=True)
    get_ch(interaction.channel_id)["active"] = True  # 答えに自動で反応できるように
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
    # DM、メンション、または会話中（/finishまで）のチャンネルで反応
    ch = get_ch(message.channel.id)
    mentioned = bot.user in message.mentions
    if not (message.guild is None or mentioned or ch["active"]):
        return
    text = message.content
    for tag in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
        text = text.replace(tag, "")
    text = text.strip()
    if mentioned and not ch["active"]:
        ch["active"] = True
        save_state()
    if not text:
        # メンションだけ → 短く返す。画像だけの投稿などには反応しない
        if mentioned:
            await message.reply(
                random.choice(CALL_REPLIES[ch["persona"]]), allowed_mentions=NO_PING
            )
        return
    async with message.channel.typing():
        try:
            answer = await chat(
                message.channel.id, message.author.id, message.author.display_name, text
            )
        except Exception as e:
            traceback.print_exc()
            await message.reply(error_text(e), allowed_mentions=NO_PING)
            return
    for i, chunk in enumerate(split_text(answer)):
        if i == 0:
            await message.reply(chunk, allowed_mentions=NO_PING)
        else:
            await message.channel.send(chunk, allowed_mentions=NO_PING)


bot.run(DISCORD_TOKEN)
