import argparse
import json
import logging
import os
import sys
from pathlib import Path
from utils import load_config, call_claude, call_gpt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_draft(draft_path: str | None) -> str:
    if draft_path:
        p = Path(draft_path)
        return p.read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    sys.exit("No draft provided.")


def build_assessment_prompt(draft: str) -> str:
    return f"""You are a rigorous peer reviewer for an AI/ML conference. You will assess the quality of the following research paper draft (abstract + plan of actions).

--- DRAFT ---
{draft}
--- END OF DRAFT ---

Evaluate the draft on the following criteria, each scored from 1 (very poor) to 10 (excellent):

- **novelty**: How original and non-obvious is the proposed research idea?
- **clarity**: How clearly and precisely is the abstract written?
- **feasibility**: How realistic and executable is the plan of actions?
- **scientific_rigor**: How technically sound and methodologically solid is the proposal?
- **impact**: How significant would the contribution be to the field?

Return your assessment as a single JSON object with this exact structure (no prose outside the JSON):

{{
  "novelty": {{
    "score": 7,
    "rationale": "Short explanation."
  }},
  "clarity": {{
    "score": 7,
    "rationale": "Short explanation."
  }},
  "feasibility": {{
    "score": 7,
    "rationale": "Short explanation."
  }},
  "scientific_rigor": {{
    "score": 7,
    "rationale": "Short explanation."
  }},
  "impact": {{
    "score": 7,
    "rationale": "Short explanation."
  }}
}}

Be objective and critical. Each rationale must be 1–2 sentences explaining the score."""


def parse_scores(raw: str) -> dict:
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rsplit("```", 1)[0].strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"Failed to parse LLM response as JSON: {e}\n\nRaw response:\n{raw}")

    return scores


def assess_draft_with_llm(
    draft_path: str, model: str, max_tokens: int, output_path: str
) -> None:
    logger.info("Loading draft ...")
    draft = load_draft(draft_path)
    logger.info("Draft loaded (%d chars)", len(draft))

    prompt = build_assessment_prompt(draft)

    logger.info("Calling %s for assessment ...", model.upper())
    if model == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY environment variable not set.")
        raw = call_claude(prompt, max_tokens)
    elif model == "gpt":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY environment variable not set.")
        raw = call_gpt(prompt, max_tokens)
    else:
        sys.exit(f"Unknown model: {model}")

    result = parse_scores(raw)

    output_str = json.dumps(result, indent=2, ensure_ascii=False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output_str, encoding="utf-8")
    logger.info("Assessment saved to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess a research paper draft using an LLM judge."
    )
    parser.add_argument(
        "--config",
        default="configs/draft_assessment.yaml",
        help="Path to YAML config (default: configs/draft_assessment.yaml)",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    model: str = cfg.get("model", "claude")
    draft_path = cfg.get("draft_path")
    output_path = cfg.get("output_path")
    max_tokens = cfg.get("max_tokens", 2048)

    assess_draft_with_llm(
        draft_path=draft_path,
        model=model,
        max_tokens=max_tokens,
        output_path=output_path,
    )
