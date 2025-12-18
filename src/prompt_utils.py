
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Optional

def load_patient_notes(subject_id: int, PATIENTS_DIR: Path, max_notes: Optional[int] = None) -> pd.DataFrame:
    """
    Load a patient's notes from subject_<ID>_notes.csv.gz.
    Only keep TEXT.
    """
    path = PATIENTS_DIR / f"subject_{subject_id}_notes.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Notes file not found: {path.resolve()}")

    df = pd.read_csv(path)

    if "TEXT" not in df.columns:
        raise ValueError("Notes file is missing TEXT column")
    
    df = df[["TEXT"]]

    if max_notes is not None:
        df = df.iloc[:max_notes]

    return df[["TEXT"]]


def load_prompt_template(PROMPT_PATH: Path) -> str:
    """
    Load the prompt template text from PROMPT_PATH.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_PATH.resolve()}")
    return PROMPT_PATH.read_text()


def make_prompt(template: str, ehr_text: str, statement: str) -> str:
    return template.format(
        ehr_text=ehr_text.rstrip(),
        statement=statement.rstrip(),
    )


def load_statements(factors_path: Path) -> List[Dict[str, str]]:
    """
    Load statement definitions from a factors JSON file.

    Only statements marked as included are returned.

    Returns a list of dicts with keys:
        - label
        - statement
    """
    if not factors_path.exists():
        raise FileNotFoundError(f"Factors file not found: {factors_path.resolve()}")

    with open(factors_path, "r") as f:
        factors_data = json.load(f)

    if "factors" not in factors_data:
        raise ValueError("Expected top-level key 'factors' in JSON file")

    factors = factors_data["factors"]

    statements = [
        {
            "label": f["label"],
            "statement": f["statement"],
        }
        for f in factors
    ]

    return statements