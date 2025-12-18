import pandas as pd
from pathlib import Path
import json

import torch
from src.llm_utils import generate_text, load_llm, safe_parse_llm_json
from src.prompt_utils import load_patient_notes, load_prompt_template, make_prompt, load_statements

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

FACTORS_PATH = Path("dtd_prompt_templates/depression_factors.json")
PATIENTS_DIR = Path("mimic-iii-demo-enriched/subject_notes")
PROMPT_PATH = Path("dtd_prompt_templates/ehr_classification_prompt.txt")

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main(max_patients: int = None, max_notes: int = None, max_new_tokens: int = 256, run_generation: bool = False):

    # Load LLM
    if run_generation:
        use_cuda = torch.cuda.is_available()
        tokenizer, model = load_llm(
            MODEL_NAME,
            load_in_4bit=use_cuda,
        )
    else:
        tokenizer = None
        model = None

    # Load statements and prompt template
    statements = load_statements(FACTORS_PATH)
    prompt_template = load_prompt_template(PROMPT_PATH)

    # Load patient notes for n example patients (if none given, use all in the directory)
    patient_files = sorted(PATIENTS_DIR.glob("subject_*_notes.csv.gz"))
    if max_patients is not None:
        patient_files = patient_files[:max_patients]

    for patient_file in patient_files:
        subject_id = int(patient_file.stem.split("_")[1])
        df_notes = load_patient_notes(subject_id, PATIENTS_DIR, max_notes=max_notes)

        # Nested JSON per patient
        patient_json = {
            "subject_id": int(subject_id),
            "notes": []
        }

        # Flat rows for per-patient CSV
        csv_rows = []

        for note_id, row in df_notes.iterrows():
            ehr_text = row["TEXT"]

            if not isinstance(ehr_text, str) or not ehr_text.strip():
                continue

            note_obj = {
                "note_id": int(note_id),
                "assessments": []
            }

            for statement_def in statements:
                label = statement_def["label"]
                statement = statement_def["statement"]

                prompt = make_prompt(prompt_template, ehr_text, statement)

                if run_generation:
                    raw_output = generate_text(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=max_new_tokens,
                    )
                    parsed = safe_parse_llm_json(raw_output)
                    conclusion = parsed["Conclusion"]
                    evidence = parsed["Evidence"]
                else:
                    conclusion = "SKIPPED"
                    evidence = ""

                assessment = {
                    "label": label,
                    "statement": statement,
                    "conclusion": conclusion,
                    "evidence": evidence,
                }
                note_obj["assessments"].append(assessment)

                csv_rows.append({
                    "subject_id": int(subject_id),
                    "note_id": int(note_id),
                    "label": label,
                    "statement": statement,
                    "conclusion": conclusion,
                    "evidence": evidence,
                })

            patient_json["notes"].append(note_obj)

        # Write per-patient JSON
        json_path = OUT_DIR / f"subject_{subject_id}_results.json"
        json_path.write_text(json.dumps(patient_json, indent=2, ensure_ascii=False))

        # Write per-patient CSV
        csv_path = OUT_DIR / f"subject_{subject_id}_results.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

        print(f"Wrote {json_path}")
        print(f"Wrote {csv_path}")
    
if __name__ == "__main__":
    
    parameters = {
        "max_patients": 1,
        "max_notes": 1,
        "run_generation": True
    }

    main(**parameters)
