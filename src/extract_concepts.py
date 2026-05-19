import argparse
import logging
import sys
import yaml
import pandas as pd
from pathlib import Path

from concept_extractors.rake import rake_extraction
from concept_extractors.yake import yake_extraction
from concept_extractors.keybert import keybert_extraction
from concept_extractors.llm import llm_extraction
from utils import load_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_results(result: pd.DataFrame, output_dir: str, file_name: str) -> None:
    """
    Save extraction results to CSV.

    The 'concepts' and 'concept_scores' columns (which are lists) are stored as string representations in the CSV.

    Args:
        result: DataFrame with extraction results.
        output_dir: Path to the output directory.
        file_name: Name of the output CSV file.
    """

    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def main(config_path: str) -> pd.DataFrame:
    """
    Main concept extraction pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # Load configuration
    config = load_config(config_path)

    # Validate input file exists
    input_file = config["general"]["input_file"]
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    methods = config["concept_extraction"]["methods"]

    # List appropriate extraction method
    extraction_methods = {
        "rake": rake_extraction,
        "yake": yake_extraction,
        "keybert": keybert_extraction,
        "llm": llm_extraction,
    }

    for method in methods:
        logger.info(f"Concept extraction method: {method}")

        if method not in extraction_methods:
            raise ValueError(
                f"Unknown extraction method: '{method}'. "
                f"Available methods: {list(extraction_methods.keys())}"
            )

        # Run extraction
        concepts = extraction_methods[method](config)

        # Save results
        output_dir = config["general"]["output_dir"]
        file_name = f"{method}_raw_concepts.csv"

        save_results(concepts, output_dir, file_name)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract concepts/keywords from scientific articles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/concept_extracting.yaml",
        help="Path to YAML configuration file (default: configs/concept_extracting.yaml)",
    )
    args = parser.parse_args()

    try:
        main(args.config)
        logger.info("Concept extraction completed successfully!")
    except Exception as e:
        logger.error(f"Concept extraction failed: {e}", exc_info=True)
        sys.exit(1)
