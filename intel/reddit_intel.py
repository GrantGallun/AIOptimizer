"""Collect and rank community evidence about local-LLM optimization.

The network-facing functions are deliberately kept separate from the scoring and
rendering functions so the latter can be exercised without Reddit access.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ARCTIC_SHIFT_API = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "AIOptimizerResearch/1.0 (contact The6Head@gmail.com)"
OUTPUT_PATH = Path(__file__).with_name("LLM_INTEL.md")

SUBS = [
    "LocalLLaMA",
    "ollama",
    "LocalLLM",
    "LLMDevs",
    "LangChain",
    "Rag",
    "PromptEngineering",
    "mlops",
    "MachineLearning",
]

KEYWORDS = [
    "inference", "quantization", "quantized", "vram", "kv cache", "kv-cache",
    "throughput", "latency", "batching", "continuous batching", "speculative",
    "context window", "num_ctx", "keep_alive", "keep alive", "flash attention",
    "gguf", "awq", "exl2", "offload", "tokens/s", "tok/s", "t/s",
    "prompt caching", "prompt cache", "semantic cache", "routing", "router",
    "agent memory", "retrieval", "embedding", "reranker", "compression",
    "llmlingua", "ollama", "vllm", "llama.cpp", "sglang", "tensorrt",
    "distillation", "draft model", "swap", "resident", "loaded model",
]

EMPIRICAL_SIGNALS = [
    "i benchmarked", "i measured", "i tested", "i profiled", "benchmark",
    "went from", "x faster", "times faster", "reduced latency", "cut latency",
    "halved", "doubled", "tripled", "tok/s", "tokens/s", "t/s", "vram usage",
    "fits in", "fits on", "quantized to", "q4", "q5", "q8", "fp8", "int4",
    "speedup", "speed up", "improved throughput", "my setup", "my rig",
    "my 3090", "my 4090", "my config",
]

# Agreement is evidence that another community member reproduced or validated a
# claim.  Intrigue captures requests for details and intent to try the technique.
AGREE_SIGNALS = [
    "agree", "confirmed", "can confirm", "works for me", "worked for me",
    "same here", "this works", "exactly", "correct", "true", "validated",
    "reproduced", "i saw the same", "i get the same", "+1",
]
INTRIGUE_SIGNALS = [
    "interesting", "curious", "how did", "how do", "can you share",
    "would you share", "more details", "tell me more", "trying this",
    "will try", "going to try", "worth trying", "what settings", "what config",
    "do you have", "any benchmarks", "why does", "wondering",
]

TIME_SECONDS = {
    "week": 7 * 24 * 60 * 60,
    "month": 30 * 24 * 60 * 60,
    "year": 365 * 24 * 60 * 60,
}


def matches(text: str | None, signals: Iterable[str]) -> bool:
    """Return whether *text* contains any signal, case-insensitively."""

    haystack = (text or "").casefold()
    return any(signal.casefold() in haystack for signal in signals)


def _children(value: Any) -> list[dict[str, Any]]:
    """Normalize list and Reddit ``data.children`` containers."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("children"), list):
        return _children(value["children"])
    data = value.get("data")
    if isinstance(data, dict) and isinstance(data.get("children"), list):
        normalized = []
        for child in data["children"]:
            if isinstance(child, dict) and isinstance(child.get("data"), dict):
                normalized.append(child["data"])
            elif isinstance(child, dict):
                normalized.append(child)
        return normalized
    return []


def _walk_comments(comments: Any) -> Iterable[dict[str, Any]]:
    for comment in _children(comments):
        yield comment
        yield from _walk_comments(comment.get("replies"))


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def score_post(post_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a scored copy of a post, including nested comment evidence."""

    scored = dict(post_dict)
    comments = list(_walk_comments(post_dict.get("comments", [])))
    agree_upvotes = 0
    intrigue_upvotes = 0

    for comment in comments:
        body = str(comment.get("body") or "")
        upvotes = max(0, _as_int(comment.get("score")))
        if matches(body, AGREE_SIGNALS):
            agree_upvotes += upvotes
        if matches(body, INTRIGUE_SIGNALS):
            intrigue_upvotes += upvotes

    post_score = _as_int(post_dict.get("score"))
    evidence_text = "\n".join(
        [str(post_dict.get("title") or ""), str(post_dict.get("selftext") or "")]
        + [str(comment.get("body") or "") for comment in comments]
    )
    scored.update(
        {
            "post_score": post_score,
            "agree_upvotes": agree_upvotes,
            "intrigue_upvotes": intrigue_upvotes,
            "community_score": post_score + 3 * agree_upvotes + 2 * intrigue_upvotes,
            "empirical": matches(evidence_text, EMPIRICAL_SIGNALS),
            "comments": comments,
        }
    )
    return scored


def _preview(text: Any, width: int = 96, limit: int = 600) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) > limit:
        clean = clean[: limit - 1].rstrip() + "…"
    return "\n".join(textwrap.wrap(clean, width=width)) or "_(no text body)_"


def render_markdown(scored: Sequence[dict[str, Any]], meta: dict[str, Any]) -> str:
    """Render ranked findings without performing I/O."""

    generated = meta.get("generated_at") or datetime.now(timezone.utc).isoformat()
    lines = [
        "# Local LLM Optimization Intel",
        "",
        f"_Generated {generated}; window={meta.get('time', 'unknown')}; "
        f"minimum post score={meta.get('min_score', 0)}._",
        "",
    ]
    ranked = sorted(scored, key=lambda item: _as_int(item.get("community_score")), reverse=True)
    for rank, post in enumerate(ranked, 1):
        empirical = " [E]" if post.get("empirical") else ""
        title = str(post.get("title") or "(untitled)")
        subreddit = str(post.get("subreddit") or "unknown")
        score = _as_int(post.get("community_score"))
        url = post.get("url") or post.get("permalink") or ""
        if isinstance(url, str) and url.startswith("/"):
            url = "https://www.reddit.com" + url
        lines.extend(
            [
                f"## {rank}. {title}{empirical}",
                "",
                f"**r/{subreddit} · community score {score} · post score "
                f"{_as_int(post.get('post_score', post.get('score')))}**",
                "",
                _preview(post.get("selftext")),
                "",
            ]
        )
        if url:
            lines.extend([f"[Open discussion]({url})", ""])
        top_comments = sorted(
            _children(post.get("comments")), key=lambda item: _as_int(item.get("score")), reverse=True
        )[:5]
        lines.extend(["**Top comments**", ""])
        if top_comments:
            for comment in top_comments:
                lines.append(f"- (+{_as_int(comment.get('score'))}) {_preview(comment.get('body'), 88, 280)}")
        else:
            lines.append("- _(none fetched)_")
        lines.extend(
            [
                "",
                f"**Agreement:** {_as_int(post.get('agree_upvotes'))} upvotes · "
                f"**Intrigue:** {_as_int(post.get('intrigue_upvotes'))} upvotes",
                "",
                "---",
                "",
            ]
        )
    if not ranked:
        lines.extend(["_No matching posts found._", ""])
    return "\n".join(lines)


def _get_json(path: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    request = Request(f"{ARCTIC_SHIFT_API}{path}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("results", []))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    return []


def fetch_posts(
    subreddit: str,
    after: int | None,
    *,
    fetch_json: Callable[[str, dict[str, Any]], Any] = _get_json,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Fetch all Arctic Shift posts in a window, newest first."""

    posts: list[dict[str, Any]] = []
    before: int | None = None
    while True:
        page = _records(
            fetch_json(
                "/posts/search",
                {"subreddit": subreddit, "after": after, "before": before, "sort": "desc", "limit": 100},
            )
        )
        if not page:
            break
        posts.extend(page)
        oldest = min(_as_int(post.get("created_utc")) for post in page)
        next_before = oldest - 1
        if len(page) < 100 or next_before < 0 or next_before == before:
            break
        before = next_before
        sleep(2)
    return posts


def fetch_comments(
    post_id: str,
    *,
    fetch_json: Callable[[str, dict[str, Any]], Any] = _get_json,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Fetch comments for one post from Arctic Shift."""

    comments: list[dict[str, Any]] = []
    before: int | None = None
    while True:
        page = _records(
            fetch_json(
                "/comments/search",
                {"link_id": post_id, "before": before, "sort": "desc", "limit": 100},
            )
        )
        if not page:
            break
        comments.extend(page)
        oldest = min(_as_int(comment.get("created_utc")) for comment in page)
        next_before = oldest - 1
        if len(page) < 100 or next_before < 0 or next_before == before:
            break
        before = next_before
        sleep(2)
    return comments


def _load_optional_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _praw_client() -> Any | None:
    """Build an optional PRAW client only when credentials are configured."""

    _load_optional_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        import praw
    except ImportError:
        return None
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=USER_AGENT)


def collect(time_window: str, min_score: int) -> list[dict[str, Any]]:
    """Collect keyword matches and attach comments to the top 30 posts."""

    after = None
    if time_window != "all":
        after = int(time.time()) - TIME_SECONDS[time_window]
    candidates: list[dict[str, Any]] = []
    first_call = True
    for subreddit in SUBS:
        if not first_call:
            time.sleep(2)
        first_call = False
        for post in fetch_posts(subreddit, after):
            post.setdefault("subreddit", subreddit)
            text = f"{post.get('title', '')}\n{post.get('selftext', '')}"
            if _as_int(post.get("score")) >= min_score and matches(text, KEYWORDS):
                candidates.append(post)

    candidates.sort(key=lambda post: _as_int(post.get("score")), reverse=True)
    for index, post in enumerate(candidates[:30]):
        if index:
            time.sleep(2)
        post["comments"] = fetch_comments(str(post.get("id") or post.get("name") or ""))
    scored = [score_post(post) for post in candidates]
    return sorted(scored, key=lambda post: post["community_score"], reverse=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", choices=("week", "month", "year", "all"), default="month")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--min-score", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    scored = collect(args.time, args.min_score)[: args.limit]
    markdown = render_markdown(scored, {"time": args.time, "min_score": args.min_score})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"wrote {len(scored)} ranked posts to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
