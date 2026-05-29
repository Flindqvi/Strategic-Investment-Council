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


async def call_openai(prompt, max_tokens=900):
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

Produce a Strategic Investment Council MVP report with this structure:

# Strategic Investment Council

## Company Summary
What does the company appear to do?

## Technology View
Does the technology appear meaningful, differentiated, scalable, or strategically relevant?

## Timing View
Why now? What broader shifts might make this company more or less relevant?

## Narrative View
How strong is the company's story, positioning, and category framing?

## Skeptic View
What are the main assumptions, gaps, weak signals, or red flags?

## Fredrik Lens
## Fredrik Lens

Assess why this company might be strategically significant through Fredrik's worldview.

Fredrik is generally interested in:

* Technology
* Timing
* Narrative
* Strategic coherence
* Category creation
* Organizational transformation
* Infrastructure and platform plays
* Governance and trust architectures (when relevant)
* Ecosystem positioning
* Network effects
* Human and AI interaction
* Long-term structural shifts

Your objective is NOT to apply a predefined theory.

Instead, identify the one or two strategic forces that are most likely to determine whether this company becomes disproportionately important over time.

Ask:

* What is the hidden driver of success?
* What is most likely to be overlooked by conventional investors?
* What future does this company assume?
* What must become true for this company to matter?

Trust, provenance, auditability, governance, and accountability are important considerations ONLY when they are genuinely relevant to the company's market, customers, business model, or strategic position.

Do not discuss them if they are not material.

Focus on the strategic force that matters most for this specific company.

## Questions Worth Exploring
List 5 questions before engaging further.

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
