import argparse
import logging
import os
import sys
from pathlib import Path
import csv
from utils import load_config, parse_concept_list, call_claude, call_gpt

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_papers(data_path: str) -> pd.DataFrame:
    """Load articles with concept lists."""
    path = Path(data_path)
    df = pd.read_csv(path)
    df["_concepts"] = df["normalized_concepts"].apply(parse_concept_list)
    return df


def find_papers_for_concept(df: pd.DataFrame, concept: str, n: int) -> pd.DataFrame:
    """Find recent n papers with particular concept."""
    c = concept.lower().strip()
    mask = df["_concepts"].apply(lambda concepts: c in concepts)
    return df[mask].sort_values("year", ascending=False).head(n)


def format_examples_block(examples: pd.DataFrame, concept: str) -> str:
    """Format article information for prompt inserting."""
    if examples.empty:
        return f'(No existing papers found for concept "{concept}" in the dataset.)\n'
    block = ""
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        block += (
            f"Example {i} ({row['year']}):\n"
            f"Title: {row['title']}\n"
            f"Abstract: {row['abstract'].strip()}\n\n"
        )
    return block.strip()


def sanitize_filename(concept: str) -> str:
    """Turn a concept string into a filesystem-friendly token."""
    return concept.strip().lower().replace(" ", "_").replace("/", "-")


def build_pair_stem(concept_a: str, concept_b: str) -> str:
    return f"{sanitize_filename(concept_a)}__x__{sanitize_filename(concept_b)}"


def load_concept_pairs(pairs_path: str) -> list[tuple[str, str]]:
    """Read concept pairs from a CSV that has columns `concept_a` and `concept_b`."""
    pairs: list[tuple[str, str]] = []
    with open(pairs_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a = row["concept_a"].strip()
            b = row["concept_b"].strip()
            if a and b:
                pairs.append((a, b))
    logger.info("Loaded %d concept pair(s) from %s", len(pairs), pairs_path)
    return pairs


def build_prompt(
    concept_a: str,
    concept_b: str,
    examples_a: pd.DataFrame,
    examples_b: pd.DataFrame,
    n_examples: int,
) -> str:
    block_a = format_examples_block(examples_a, concept_a)
    block_b = format_examples_block(examples_b, concept_b)

    return f"""You are an expert AI/ML researcher. Your task is to propose a novel research paper that bridges the following two concepts:

Concept A: {concept_a}
Concept B: {concept_b}

Below are up to {n_examples} recent papers for each concept separately. Use them to understand the current state of each research area and identify a clear gap at their intersection.

--- PAPERS ON CONCEPT A: {concept_a} ---
{block_a}
--- END OF PAPERS ON CONCEPT A ---

--- PAPERS ON CONCEPT B: {concept_b} ---
{block_b}
--- END OF PAPERS ON CONCEPT B ---

Based on the above, produce exactly two sections:

1. **Abstract** (150–200 words): A self-contained summary of the proposed paper — motivation, core idea, method sketch, and expected contribution. Do not repeat the existing papers verbatim; this must describe a novel work.

2. **Plan of Actions** (bullet list, 6–10 items): Concrete steps to execute this research — e.g. dataset collection, model design, baseline selection, evaluation protocol, ablation study, writing. Keep each item to one sentence.

Use academic writing style. Be specific and technical."""


def generate_draft(
    data_path: str,
    concept_a: str,
    concept_b: str,
    model: str,
    n_examples: int,
    max_tokens: int,
    drafts_dir: str,
) -> None:
    logger.info("Loading papers from %s", data_path)
    df = load_papers(data_path)

    logger.info("Searching for papers with '%s'", concept_a)
    examples_a = find_papers_for_concept(df, concept_a, n_examples)
    logger.info("Searching for papers with '%s'", concept_b)
    examples_b = find_papers_for_concept(df, concept_b, n_examples)

    prompt = build_prompt(concept_a, concept_b, examples_a, examples_b, n_examples)

    logger.info("Calling %s ...", model.upper())
    if model == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY environment variable not set.")
        result = call_claude(prompt, max_tokens)
    elif model == "gpt":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY environment variable not set.")
        result = call_gpt(prompt, max_tokens)
    else:
        sys.exit(f"Unknown model: {model}")

    # Build output paths from drafts_dir and the concept pair
    pair_stem = build_pair_stem(concept_a, concept_b)
    drafts = Path(drafts_dir)
    drafts.mkdir(parents=True, exist_ok=True)

    output_path = drafts / f"{pair_stem}_draft.txt"
    prompt_output_path = drafts / f"{pair_stem}_prompt.txt"

    output_path.write_text(result, encoding="utf-8")
    logger.info("Draft saved to %s", output_path)

    prompt_output_path.write_text(prompt, encoding="utf-8")
    logger.info("Prompt saved to %s", prompt_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a paper abstract + action plan for a concept pair."
    )
    parser.add_argument(
        "--config",
        default="configs/draft_generation.yaml",
        help="Path to YAML config (default: configs/draft_generation.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    concept_a: str = cfg["concept_a"]
    concept_b: str = cfg["concept_b"]
    model: str = cfg.get("model", "claude")
    n_examples: int = cfg.get("n_examples", 3)
    data_path: str = cfg.get(
        "data_path", "data/interim/llm/normalized_concepts_all.csv"
    )
    max_tokens = cfg.get("max_tokens", 2048)

    pairs_path: str = cfg["pairs_path"]
    drafts_dir: str = cfg.get("drafts_dir", "drafts")

    pairs = load_concept_pairs(pairs_path)

    for idx, (concept_a, concept_b) in enumerate(pairs, start=1):
        logger.info(
            "=== Pair %d/%d: '%s' × '%s' ===", idx, len(pairs), concept_a, concept_b
        )
        generate_draft(
            data_path=data_path,
            concept_a=concept_a,
            concept_b=concept_b,
            model=model,
            n_examples=n_examples,
            max_tokens=max_tokens,
            drafts_dir=drafts_dir,
        )

    logger.info("All %d pair(s) processed. Drafts are in '%s/'", len(pairs), drafts_dir)
