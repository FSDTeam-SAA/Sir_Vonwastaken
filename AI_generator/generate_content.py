"""
Module 7 — AI Content Generator orchestration.

Calls each prompts.py function as its OWN LLM request (per-feature, not
one combined call) — see the conversation/README note on why: it lets a
creator regenerate just titles, or just thumbnails, without redoing
everything else, and lets each content type use its own temperature.

Trade-off vs. a single combined call: more API calls (6 vs 1) and later
pieces don't automatically see earlier ones unless you thread context
forward — handled here by passing the chosen title into hooks/thumbnails,
and the chosen idea/outline into outline/script.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from AI_generator import prompts
from creator_profile.build_creator_profile import get_creator_profile_summary
from database.mongodb import find_one, get_db
from utils.llm_client import chat_complete, chat_complete_json
from utils.logger import logger

# Per-type generation settings — tune independently, unlike a single combined call
_GEN_SETTINGS = {
    "titles": {"max_tokens": 300, "temperature": 0.8},
    "hooks": {"max_tokens": 300, "temperature": 0.8},
    "video_ideas": {"max_tokens": 400, "temperature": 0.9},
    "outline": {"max_tokens": 500, "temperature": 0.4},
    "script": {"max_tokens": 900, "temperature": 0.7},
    "thumbnails": {"max_tokens": 500, "temperature": 0.7},
}


def _creator_context_str(channel_id: Optional[str]) -> str:
    if not channel_id:
        return ""
    profile = get_creator_profile_summary(channel_id) or {}
    if not profile:
        return ""
    return f"Categories: {', '.join(profile.get('top_categories', []))}; Topics: {', '.join(profile.get('top_topics', []))}"


# ---------------------------------------------------------------------------
# Individually-callable generators (each = one independent LLM call)
# ---------------------------------------------------------------------------

def gen_titles(theme: str, channel_id: Optional[str] = None) -> List[str]:
    prompt = prompts.generate_titles(theme, creator_context=_creator_context_str(channel_id))
    result = chat_complete_json(prompts.CONTENT_STRATEGIST_SYSTEM_PROMPT, prompt, **_GEN_SETTINGS["titles"])
    return result.get("titles", [])


def gen_hooks(topic: str, tone: str = "engaging") -> List[str]:
    prompt = prompts.generate_hooks(topic, tone=tone)
    result = chat_complete_json(prompts.CONTENT_STRATEGIST_SYSTEM_PROMPT, prompt, **_GEN_SETTINGS["hooks"])
    return result.get("hooks", [])


def gen_video_ideas(niche: str, trend_summary: str = "") -> List[str]:
    prompt = prompts.video_ideas(niche, trend_summary=trend_summary)
    result = chat_complete_json(prompts.CONTENT_STRATEGIST_SYSTEM_PROMPT, prompt, **_GEN_SETTINGS["video_ideas"])
    return result.get("video_ideas", [])


def gen_outline(video_idea: str) -> List[Dict]:
    prompt = prompts.outlines(video_idea)
    result = chat_complete_json(prompts.CONTENT_STRATEGIST_SYSTEM_PROMPT, prompt, **_GEN_SETTINGS["outline"])
    return result.get("outline", [])


def gen_script(script_idea: str, outline: Optional[List[Dict]] = None) -> str:
    outline_text = ""
    if outline:
        outline_text = "\n".join(f"- {s.get('section')}: {s.get('description')}" for s in outline)
    prompt = prompts.draft_script(script_idea, outline=outline_text)
    return chat_complete(
        prompts.CONTENT_STRATEGIST_SYSTEM_PROMPT, prompt, **_GEN_SETTINGS["script"]
    ).strip()


def gen_thumbnails(title: str, video_idea: str = "") -> List[Dict]:
    prompt = prompts.thumbnail_suggestions(title, video_idea=video_idea)
    result = chat_complete_json(prompts.CONTENT_STRATEGIST_SYSTEM_PROMPT, prompt, **_GEN_SETTINGS["thumbnails"])
    return result.get("thumbnail_ideas", [])


# ---------------------------------------------------------------------------
# Full-package orchestration (calls all 6, threading context forward)
# ---------------------------------------------------------------------------

def generate_content_for_trend(content_id: str, channel_id: Optional[str] = None) -> Dict:
    """
    Looks up a ranked trend by content_id (from `trend_candidates`) and
    generates the full content package by calling each generator in
    sequence, feeding earlier outputs into later prompts for consistency:
    titles -> (pick first) -> hooks + outline -> script -> thumbnails.
    """
    trend = find_one("trend_candidates", {"content_id": content_id})
    if not trend:
        raise ValueError(f"No trend_candidate found with content_id={content_id}. Run the pipeline first.")

    channel_id = channel_id or trend.get("channel_id")
    theme = trend.get("title", "")

    titles = gen_titles(theme, channel_id=channel_id)
    chosen_title = titles[0] if titles else theme

    hooks = gen_hooks(chosen_title)
    outline = gen_outline(chosen_title)
    script = gen_script(chosen_title, outline=outline)
    thumbnails = gen_thumbnails(chosen_title, video_idea=theme)

    doc = {
        "trend_id": content_id,
        "channel_id": channel_id,
        "trend_title": trend.get("title"),
        "titles": titles,
        "hooks": hooks,
        "outline": outline,
        "script_draft": script,
        "thumbnail_ideas": thumbnails,
        "generated_at": datetime.utcnow(),
    }
    get_db().generated_content.insert_one(doc.copy())
    doc.pop("_id", None)

    logger.info(f"Generated content package for trend '{trend.get('title')}'")
    return doc


def regenerate_field(content_id: str, field: str, channel_id: Optional[str] = None) -> Dict:
    """
    Regenerates ONE field of an already-generated content package (the
    main payoff of the per-feature prompt design) and updates the stored
    doc in place. `field` is one of: titles, hooks, outline, script, thumbnails.
    """
    doc = get_db().generated_content.find_one({"trend_id": content_id}, sort=[("generated_at", -1)])
    if not doc:
        raise ValueError(f"No generated content found for content_id={content_id}. Run generate_content_for_trend first.")

    theme = doc.get("trend_title", "")
    chosen_title = (doc.get("titles") or [theme])[0]

    if field == "titles":
        new_value = gen_titles(theme, channel_id=channel_id)
    elif field == "hooks":
        new_value = gen_hooks(chosen_title)
    elif field == "outline":
        new_value = gen_outline(chosen_title)
    elif field == "script":
        new_value = gen_script(chosen_title, outline=doc.get("outline"))
    elif field == "thumbnails":
        new_value = gen_thumbnails(chosen_title, video_idea=theme)
    else:
        raise ValueError(f"Unknown field '{field}'. Must be one of: titles, hooks, outline, script, thumbnails")

    field_key = "script_draft" if field == "script" else ("thumbnail_ideas" if field == "thumbnails" else field)
    get_db().generated_content.update_one(
        {"_id": doc["_id"]}, {"$set": {field_key: new_value, "updated_at": datetime.utcnow()}}
    )
    logger.info(f"Regenerated '{field}' for trend_id={content_id}")
    return {field_key: new_value}


def get_generated_content_history(channel_id: str, limit: int = 20):
    from database.mongodb import find

    return find("generated_content", {"channel_id": channel_id}, limit=limit, sort=[("generated_at", -1)])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m AI_generator.generate_content <content_id>")
        sys.exit(1)
    print(generate_content_for_trend(sys.argv[1]))