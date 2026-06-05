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
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

MODEL = "gpt-4.1-mini"
MAX_CHUNK = 1800
MAX_SITE_CHARS = 12000
MAX_EXTERNAL_CHARS = 6000

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

def tinyfish_get(endpoint, params):
    if not TINYFISH_API_KEY:
        raise RuntimeError("TINYFISH_API_KEY is not set")

    headers = {
        "X-API-Key": TINYFISH_API_KEY
    }

    response = requests.get(endpoint, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()

def extract_text_from_tinyfish_response(data):
    if isinstance(data, list) and data:
        data = data[0]

    if isinstance(data, dict):
        for key in ["markdown", "text", "content", "result"]:
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, list) and value:
                return extract_text_from_tinyfish_response(value)

            if isinstance(value, dict):
                return extract_text_from_tinyfish_response(value)

        for value in data.values():
            if isinstance(value, str) and len(value) > 200:
                return value.strip()

    return str(data)[:MAX_SITE_CHARS]

def tinyfish_fetch_text(url):
    if not TINYFISH_API_KEY:
        raise RuntimeError("TINYFISH_API_KEY is not set")

    headers = {
        "X-API-Key": TINYFISH_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://api.fetch.tinyfish.ai",
        json={
            "urls": [url],
            "format": "markdown"
        },
        headers=headers,
        timeout=150
    )

    response.raise_for_status()
    data = response.json()

    return extract_text_from_tinyfish_response(data)[:MAX_SITE_CHARS]

def fetch_best_website_text(url):
    try:
        return tinyfish_fetch_text(url)
    except Exception as e:
        print(f"TinyFish fetch failed, using fallback scraper: {e}")
        return fetch_website_text(url)


def extract_search_results(data, limit=3):
    results = []

    if isinstance(data, dict):
        possible_lists = []

        for key in ["results", "items", "data"]:
            value = data.get(key)
            if isinstance(value, list):
                possible_lists = value
                break

        if not possible_lists and isinstance(data.get("result"), list):
            possible_lists = data.get("result")

        for item in possible_lists[:limit]:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or "Untitled"
                url = item.get("url") or item.get("link") or ""
                snippet = item.get("snippet") or item.get("description") or item.get("text") or ""
                results.append(f"- {title}\n  {url}\n  {snippet}")

    return "\n".join(results)


def tinyfish_search(query, limit=3):
    data = tinyfish_get(
        "https://api.search.tinyfish.ai",
        {"query": query}
    )

    return extract_search_results(data, limit=limit)

def get_company_hint(url):
    domain = re.sub(r"^https?://", "", url)
    domain = domain.split("/")[0]
    domain = domain.replace("www.", "")
    return domain.split(".")[0]


def research_company(url):
    company_hint = get_company_hint(url)

    queries = [
        f"{company_hint} competitors",
        f"{company_hint} market trends",
        f"{company_hint} funding news founder"
    ]

    sections = []

    for query in queries:
        try:
            results = tinyfish_search(query, limit=3)
            if results.strip():
                sections.append(f"Search query: {query}\n{results}")
        except Exception as e:
            print(f"TinyFish search failed for '{query}': {e}")

    if not sections:
        return "External research unavailable. Continue with website-only analysis."

    external_context = "\n\n".join(sections)
    return external_context[:MAX_EXTERNAL_CHARS]

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
        site_text = await asyncio.to_thread(fetch_best_website_text, url)
        external_context = await asyncio.to_thread(research_company, url)

        prompt = f"""

Analyze this company based only on the website text below.
URL:
{url}

Website text:
{site_text}

External research:
{external_context}

Use external research mainly to improve Timing View, Platform View, Skeptic View, and Fredrik Lens.

Clearly distinguish company claims from external signals.

Do not over-weight weak or irrelevant search results.

Before beginning the analysis:

Determine whether the available information is sufficient to identify:

Industry
Customer
Product

with reasonable confidence.

Special rule for acronyms and domain-specific terminology:

If understanding the company depends on an acronym, abbreviation, or specialized term that is not clearly explained:

Do not assume a meaning based on prior knowledge.
Treat confidence as reduced.
Explicitly acknowledge the ambiguity.
Prefer uncertainty over selecting a single interpretation.

Ask yourself:

"Could this analysis change materially if I have misunderstood a key acronym, abbreviation, or domain-specific term?"

If yes, confidence should normally be considered low unless the website clearly provides additional context.

If confidence is reasonably high:

Proceed directly to the Strategic Investment Council analysis.
Do not mention confidence levels, evidence quality, or observations.

If confidence is low:

Include before the analysis:

Limited Information Warning

Briefly explain why the available information is insufficient.

Observed Facts

List 3-5 concrete observations directly supported by the website.

When confidence is low:

Reduce strategic speculation.
Frame conclusions as hypotheses rather than observations.
Focus on uncertainty reduction.
The Investment Question should prioritize resolving the key uncertainty.

General rules:

Do not infer industry from company names, domain names, acronyms, abbreviations, or branding alone.
Do not assume a business model, customer, market, or technology that is not supported by evidence.
If multiple interpretations are possible, acknowledge the ambiguity.
Prefer uncertainty over incorrect specificity.
Every major strategic conclusion should be traceable to evidence found on the website.

You are the Strategic Investment Council.

The objective is NOT to merely summarize the company.

The objective is to identify the few variables that will determine whether this company becomes strategically important.

Focus on judgment, not coverage.

Avoid generic startup analysis, MBA language, feature summaries, and long lists.

Prefer identifying hidden strategic variables over describing the company.

Keep the report concise, insightful, and high-signal.

Use the following structure:

#Strategic Investment Council
##Company Summary

In 3-5 sentences:

What does the company appear to do?
Who appears to be the customer?
What problem appears to be solved?

Do not repeat this information later.

##Origin Insight

Ask:

What non-obvious insight may have led to this company?
What frustration, inefficiency, contradiction, or market gap does the company appear to have noticed?
What might the founders understand that others overlook?

Do not discuss founder biographies.

Focus on the underlying insight that gave rise to the company.

##Timing View

Ask:

Why now?
What structural shifts make this company more relevant today than five years ago?
What has recently become possible, necessary, or inevitable?

Focus on timing, not market size.

##Platform View

Ask:

Where does power accumulate?
What becomes stronger as the company grows?
What is the single most important source of leverage?
What could become difficult for competitors to replicate?

Identify the primary source of defensibility.

Do not list multiple moats.

Focus on leverage, strategic position, and power concentration.

##Skeptic View

Identify the single assumption most likely to be wrong.

Ask:

"What must be true for this company to succeed?"

Then challenge that assumption directly.

Be specific.

Be intellectually honest.

Avoid generic startup skepticism.

##Fredrik Lens

Assess why this company might become strategically significant.

Do not assume the company is creating a new category.

First determine whether the company is primarily:

- A Category Participant
- A Category Improver
- A Category Consolidator
- A Category Leader
- An Infrastructure Layer
- A Category Creator

Only discuss category creation when there is strong evidence that the company is changing how the market defines the problem itself.

The most strategically important companies are not necessarily category creators. Many become valuable by owning critical workflows, becoming trusted infrastructure, concentrating data, reducing friction, or embedding themselves deeply into existing systems.

Distinguish between:

What the company sells
What strategic position it is trying to occupy

Fredrik is generally interested in:

Timing
Narrative
Category creation
Strategic coherence
Platform and ecosystem dynamics
Network effects
Human and AI interaction
Organizational transformation
Long-term structural shifts
Governance and trust architectures (only when materially relevant)

Do not apply a predefined theory.

Instead identify the one or two strategic forces that matter most.

Ask:

What is the hidden driver of success?
What are conventional investors likely to miss?
What future does this company assume?
What must become true for this company to matter?
What game is this company actually playing?
What behavior, belief, or assumption is the company trying to change?
If this company wins, what new idea becomes obvious in hindsight?

Narrative is not marketing.

Narrative is a mechanism for changing how customers, markets, industries, or investors understand reality.

When relevant, identify the narrative that could make the company disproportionately important.

Do not force discussion of:

Trust
Provenance
Auditability
Governance
Accountability

These should only be discussed when genuinely relevant.

Focus on the strategic force that matters most for this company.

##Investment Question

What is the single most important unanswered question that would most improve judgment?

Choose the question with the highest information value.

##Verdict

Maximum 3 sentences.

Format:

Interesting if:
...

Uninteresting if:
...

Ultimately this company is betting on:
...

Avoid generic conclusions.

Make the verdict reflect the central strategic variable identified in the analysis.

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

