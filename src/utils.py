import yaml
from typing import Any, Dict, List, Optional, Set
import ast


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def parse_concept_list(val) -> List[str]:
    return [str(concept).strip() for concept in ast.literal_eval(val) if concept]
