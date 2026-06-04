import os
import re
import asyncio
import requests
import discord
from bs4 import BeautifulSoup
from discord.ext import commands
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

MODEL = "gpt-4.1-mini"
MAX_CHUNK = 1800
MAX_SITE_CHARS = 12000


def extract_url(text):
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else None


def fetch_website_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 StrategicInvestmentCouncilBot/0.1"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:MAX_SITE_CHARS]


def split_message(text, chunk_size=MAX_CHUNK):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


async def send_long_message(channel, text):
    for chunk in split_message(text):
        await channel.send(chunk)


async def call_openai(prompt, max_tokens=1500):
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are the Strategic Investment Council.

The objective is not confirmation. The objective is better judgment.

You are not a generic MBA-style startup analyst.
You evaluate startups through a strategic lens focused on:
technology, timing, narrative, strategic coherence, structural shifts,
second-order effects, and why a company might matter in the future.

Do not provide investment advice. Provide strategic analysis.
Be concise, sharp, and thoughtful.
"""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens
    )

    finish_reason = response.choices[0].finish_reason
    print(f"Finish reason: {finish_reason}")

    return response.choices[0].message.content.strip()

async def analyze_url(channel, url):
    thinking = await channel.send(f"Analyzing {url} through the Strategic Investment Council...")

    try:
        site_text = await asyncio.to_thread(fetch_website_text, url)

        prompt = f"""
Analyze this company based only on the website text below.
URL:
{url}
Website text:
{site_text}

You are the Strategic Investment Council.

The objective is NOT to merely summarize the company.

The objective is to identify the few variables that will determine whether this company becomes strategically important.

Focus on judgment, not coverage.

Avoid generic startup analysis, MBA language, and feature summaries.

Each council member must contribute a distinct perspective.

Keep the report concise, insightful, and high-signal.

Use the following structure:

# Strategic Investment Council

## Company Summary

In 3-5 sentences:

- What does the company appear to do?
- Who appears to be the customer?
- What problem appears to be solved?

Do not repeat this information later.

---

## Founder View

Ask:

- Why might this team be uniquely positioned to win?
- What founder insight or experience may have led to this company?
- Is there evidence of founder-market fit?

If information is limited, state what would need to be true.

Focus on founder advantage, not biography.

---

## Market View

Ask:

- Why now?
- What broader structural shifts make this company more relevant?
- Is demand being pulled by the market, or pushed by the company?
- If successful, how large could the opportunity become?

Focus on timing and market inevitability.

---

## Platform View

Ask:

- Where does defensibility come from?
- What becomes stronger as the company grows?
- Are there network effects, ecosystem effects, switching costs, proprietary data, infrastructure advantages, or platform dynamics?
- Where does power accumulate?

Do not focus on features.

Focus on moats and strategic leverage.

---

## Skeptic View

Do NOT provide a list of generic risks.

Identify the single assumption most likely to be wrong.

Ask:

"What must be true for this company to succeed?"

Then challenge that assumption directly.

Be sharp, specific, and intellectually honest.

---

## Fredrik Lens

Assess why this company might become strategically significant.

Fredrik is generally interested in:

- Technology
- Narrative
- Category creation
- Strategic coherence
- Platform and infrastructure plays
- Ecosystem positioning
- Network effects
- Human and AI interaction
- Organizational transformation
- Long-term structural shifts
- Governance and trust architectures (ONLY when materially relevant)

Do NOT apply a predefined theory.

Instead identify the one or two strategic forces that matter most.

Ask:

- What is the hidden driver of success?
- What are conventional investors likely to miss?
- What future does this company assume?
- What must become true for this company to matter?
- Is this a category participant, category leader, category creator, or infrastructure layer?

Trust, provenance, auditability, governance, and accountability should ONLY be discussed when they are genuinely relevant to the company's future success.

Do not force them into the analysis.

Focus on the strategic force that matters most for this specific company.

---

## Investment Question

What is the single most important unanswered question that would most improve judgment?

Only one question.

Choose the question with the highest information value.

---

## Verdict

Maximum 3 sentences.

Format:

Interesting if:
...

Uninteresting if:
...

One sentence on what the company is ultimately betting on.

---

End with:

"The objective is not confirmation. The objective is better judgment."
"""

        report = await call_openai(prompt)

        await thinking.delete()
        await send_long_message(channel, report)

    except Exception as e:
        await thinking.delete()
        await channel.send(f"Could not analyze the URL. Error: {str(e)}")


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    content = message.content.strip()
    lower = content.lower()

    if lower.startswith("analyze "):
        url = extract_url(content)
        if not url:
            await message.channel.send("Please include a valid URL.")
            return

        await analyze_url(message.channel, url)
        return

    await bot.process_commands(message)


@bot.command()
async def analyze(ctx, url):
    await analyze_url(ctx.channel, url)


@bot.command()
async def helpme(ctx):
    await ctx.send("""
Strategic Investment Council MVP

Use:

Analyze https://company.com

or:

/analyze https://company.com

The objective is not confirmation. The objective is better judgment.
""")


bot.run(DISCORD_TOKEN)
