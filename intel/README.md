# intel/ — external signal feeds for LANDSCAPE + gateway stages

Pipeline: scrape practitioner sources -> ranked digest -> Fable distills into LANDSCAPE.md ->
actionable claims become receipts-gated PRODUCT.md stages (example: the r/LocalLLaMA swap-thrash
thread -> measured 32.3% reload overhead in our own runs -> keep_alive fix + stage 8).

- `reddit_intel.py` — adapted from the user's reddit_hypothesis.py (creator-strategy scanner from
  another project; scoring design preserved: community_score = upvotes + 3*agree + 2*intrigue,
  plus empirical-claim flags). Arctic Shift archive (no credentials needed) or PRAW when
  REDDIT_CLIENT_ID/SECRET are set.
- Candidate next feeds: Hacker News via the Algolia API (clean JSON, no auth), arXiv cs.CL/cs.LG
  new-submissions RSS, HuggingFace blog/forums.
