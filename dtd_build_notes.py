from pathlib import Path
from typing import Optional, Sequence, Set

import pandas as pd


DEMO_DIR = Path("mimic-iii-clinical-database-demo-1.4")
FULL_DIR = Path("mimic-iii-clinical-database-1.4")

ENRICHED_DIR = Path("mimic-iii-demo-enriched")
NOTES_OUT = ENRICHED_DIR / "NOTEEVENTS.csv.gz"
CHART_OUT = ENRICHED_DIR / "CHARTEVENTS.csv.gz"

SUBJECT_NOTES_DIR = Path("subject_notes")


def load_demo_tables(demo_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load the MIMIC-III demo CSV tables into a dict keyed by uppercase table name.
    """
    tables: dict[str, pd.DataFrame] = {}
    for csv_path in demo_dir.glob("*.csv"):
        name = csv_path.stem.upper()
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.upper()
        tables[name] = df
    return tables


def extract_demo_rows(
    full_dir: Path,
    table_filename: str,
    demo_subject_ids: Set[int],
    demo_hadm_ids: Set[int],
    demo_icustay_ids: Optional[Set[int]] = None,
    usecols: Optional[Sequence[str]] = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Load only rows from a large MIMIC-III table that belong to the demo cohort.

    This function streams the full table in chunks and keeps any row whose
    SUBJECT_ID, HADM_ID, or ICUSTAY_ID belongs to the demo cohort.
    """
    path = full_dir / table_filename

    iterator = pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        engine="python",
        chunksize=chunksize,
    )

    kept_parts: list[pd.DataFrame] = []

    for chunk in iterator:
        masks: list[pd.Series] = []

        if "SUBJECT_ID" in chunk.columns:
            masks.append(chunk["SUBJECT_ID"].isin(demo_subject_ids))

        if "HADM_ID" in chunk.columns:
            masks.append(chunk["HADM_ID"].isin(demo_hadm_ids))

        if demo_icustay_ids is not None and "ICUSTAY_ID" in chunk.columns:
            masks.append(chunk["ICUSTAY_ID"].isin(demo_icustay_ids))

        if not masks:
            continue

        mask = masks[0]
        for m in masks[1:]:
            mask |= m

        filtered = chunk[mask]
        if not filtered.empty:
            kept_parts.append(filtered)

    if not kept_parts:
        return pd.DataFrame(columns=list(usecols) if usecols is not None else None)

    return pd.concat(kept_parts, ignore_index=True)


def write_subject_note_files(demo_notes: pd.DataFrame, out_dir: Path) -> None:
    """
    Write one compressed CSV per SUBJECT_ID from the demo NOTEEVENTS subset.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    for subject_id, df_sub in demo_notes.groupby("SUBJECT_ID"):
        out_path = out_dir / f"subject_{subject_id}_notes.csv.gz"
        df_sub.to_csv(out_path, index=False, compression="gzip")


def build_or_load_enriched(
    demo_dir: Path,
    full_dir: Path,
    enriched_dir: Path,
    notes_out: Path,
    chart_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load enriched demo extracts if present; otherwise create them from the full tables.
    """
    enriched_dir.mkdir(exist_ok=True, parents=True)

    if notes_out.exists() and chart_out.exists():
        print("Enriched files already exist. Loading from disk...")
        demo_notes = pd.read_csv(notes_out)
        demo_chart = pd.read_csv(chart_out)
        return demo_notes, demo_chart

    print("Enriched files not found. Running extraction from full MIMIC-III tables...")

    demo_tables = load_demo_tables(demo_dir)
    adm_demo = demo_tables["ADMISSIONS"]
    icu_demo = demo_tables["ICUSTAYS"]

    demo_subject_ids = set(adm_demo["SUBJECT_ID"])
    demo_hadm_ids = set(adm_demo["HADM_ID"])
    demo_icustay_ids = set(icu_demo["ICUSTAY_ID"])

    demo_notes = extract_demo_rows(
        full_dir=full_dir,
        table_filename="NOTEEVENTS.csv.gz",
        demo_subject_ids=demo_subject_ids,
        demo_hadm_ids=demo_hadm_ids,
        usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
    )
    print(f"NOTEEVENTS extracted: {demo_notes.shape}")

    demo_chart = extract_demo_rows(
        full_dir=full_dir,
        table_filename="CHARTEVENTS.csv.gz",
        demo_subject_ids=demo_subject_ids,
        demo_hadm_ids=demo_hadm_ids,
        demo_icustay_ids=demo_icustay_ids,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
    )
    print(f"CHARTEVENTS extracted: {demo_chart.shape}")

    demo_notes.to_csv(notes_out, index=False, compression="gzip")
    demo_chart.to_csv(chart_out, index=False, compression="gzip")
    print("Extraction finished and saved.")

    return demo_notes, demo_chart


def main() -> None:
    demo_notes, demo_chart = build_or_load_enriched(
        demo_dir=DEMO_DIR,
        full_dir=FULL_DIR,
        enriched_dir=ENRICHED_DIR,
        notes_out=NOTES_OUT,
        chart_out=CHART_OUT,
    )

    # Quick sanity check output
    print("Demo notes preview:")
    print(demo_notes.head())

    print("\nDemo chart preview:")
    print(demo_chart.head())

    # Write per-subject notes
    write_subject_note_files(demo_notes, SUBJECT_NOTES_DIR)
    print(f"\nPer-subject note files written to: {SUBJECT_NOTES_DIR.resolve()}")


if __name__ == "__main__":
    main()