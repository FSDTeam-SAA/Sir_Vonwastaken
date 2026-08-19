"""
AI_generator/prompts.py

Each function returns a STRING PROMPT (no LLM call happens here — that's
done in generate_content.py). Model used: configured via
OPENAI_CHAT_MODEL in .env (default gpt-4o-mini; swap to whatever GPT-5
mini's exact model string is once you confirm it in your OpenAI dashboard
-- "GPT 5 mini" wasn't in the model list I can verify from here, so
double check the exact string before deploying).

NOTE ON PARAMS: your original template left most function signatures as
placeholders (only `generate_titles(theme)` had a real param). I filled
in the rest with what each generator actually needs to produce a good
result — flagged inline below. Swap/rename freely; the important part is
each one stays a single-purpose, independently-callable prompt builder,
matching your template's intent (so a creator can regenerate just the
titles, just the hooks, etc. without re-running everything).
"""

# Shared system prompt reused across all calls in generate_content.py
CONTENT_STRATEGIST_SYSTEM_PROMPT = """You are an expert YouTube content strategist and scriptwriter who helps \
creators turn trending topics into videos that match their established channel style and tone."""


def generate_titles(theme: str, creator_context: str = "") -> str:
    """theme: the trending topic/theme to generate titles for.
    creator_context: optional — creator's top categories/topics/tone, for style matching."""
    context_block = f"\nCreator's usual style/topics: {creator_context}" if creator_context else ""
    return f"""Generate 5 YouTube video title options for this theme: "{theme}"{context_block}

Rules:
- Each title under 60 characters
- Clickable but not misleading clickbait
- Vary the angle across the 5 (curiosity, how-to, listicle, contrarian, story-driven)

Respond with a JSON object: {{"titles": ["title 1", "title 2", "title 3", "title 4", "title 5"]}}
Output ONLY the JSON, no prose."""


def generate_hooks(topic: str, tone: str = "engaging") -> str:
    """topic: the video's core topic/title chosen.
    tone: optional tone descriptor (e.g. 'engaging', 'urgent', 'calm/informative')."""
    return f"""Generate 3 opening hooks (first 5-10 spoken seconds) for a YouTube video about: "{topic}"
Tone: {tone}

Rules:
- Each hook must create curiosity or tension in the very first line
- Written as spoken dialogue, not a title
- No generic openers like "Hey guys, welcome back"

Respond with a JSON object: {{"hooks": ["hook 1", "hook 2", "hook 3"]}}
Output ONLY the JSON, no prose."""


def video_ideas(niche: str, trend_summary: str = "") -> str:
    """niche: creator's content niche/category.
    trend_summary: optional — summary of the trending topic driving this idea generation."""
    trend_block = f"\nBased on this trending topic: {trend_summary}" if trend_summary else ""
    return f"""Generate 5 concrete YouTube video idea concepts for a creator in the "{niche}" niche.{trend_block}

Each idea should be a single sentence describing a specific, filmable video — not a vague topic.

Respond with a JSON object: {{"video_ideas": ["idea 1", "idea 2", "idea 3", "idea 4", "idea 5"]}}
Output ONLY the JSON, no prose."""


def outlines(video_idea: str) -> str:
    """video_idea: the chosen video concept/title to build a structural outline for."""
    return f"""Create a video outline for this video idea: "{video_idea}"

Break it into 4-7 sections. Each section needs a name and a 1-sentence description of what happens in it.

Respond with a JSON object:
{{"outline": [{{"section": "Intro", "description": "..."}}, {{"section": "...", "description": "..."}}]}}
Output ONLY the JSON, no prose."""


def draft_script(script_idea: str, outline: str = "") -> str:
    """script_idea: the video concept/title.
    outline: optional — the outline (as text) to follow, for consistency with a prior outlines() call."""
    outline_block = f"\nFollow this outline:\n{outline}" if outline else ""
    return f"""Write a full first-draft YouTube script for: "{script_idea}"{outline_block}

Rules:
- 300-500 words
- Natural spoken-language pacing, not written-essay style
- Use blank lines between sections
- Plain text only — do not wrap in JSON or markdown fences

Write the script now."""


def thumbnail_suggestions(title: str, video_idea: str = "") -> str:
    """title: the chosen video title.
    video_idea: optional — fuller video concept for extra visual context."""
    context_block = f"\nVideo concept: {video_idea}" if video_idea else ""
    return f"""Generate 3 YouTube thumbnail concepts for a video titled: "{title}"{context_block}

For each concept provide: a short concept description, 3-5 word text overlay for the thumbnail,
and visual notes (composition/color/emotion).

Respond with a JSON object:
{{"thumbnail_ideas": [{{"concept": "...", "text_overlay": "...", "visual_notes": "..."}}]}}
Output ONLY the JSON, no prose."""