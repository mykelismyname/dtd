# %% [markdown]
# ## Load data
# ### General imports and setup

# %% [markdown]
# Imports and configuration.

# %%
import pandas as pd
from pathlib import Path
from textwrap import fill
from collections import Counter

DEMO_DIR      = Path("mimic-iii-clinical-database-demo-1.4")
FULL_DIR      = Path("mimic-iii-clinical-database-1.4")
ENRICHED_DIR  = Path("mimic-iii-demo-enriched")
ENRICHED_DIR.mkdir(exist_ok=True)

NOTES_OUT = ENRICHED_DIR / "NOTEEVENTS.csv.gz"
CHART_OUT = ENRICHED_DIR / "CHARTEVENTS.csv.gz"

# %% [markdown]
# Functions definitions.

# %%
def extract_demo_rows(
    full_dir: Path,
    table_filename: str,
    demo_subject_ids: set,
    demo_hadm_ids: set,
    demo_icustay_ids: set = None,
    usecols=None,
    chunksize=50_000,
):
    """
    Load only rows from a large MIMIC-III table that belong to the demo cohort.
    Works for NOTEEVENTS and CHARTEVENTS.
    """
    path = full_dir / table_filename
    
    iterator = pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        engine="python",          # tolerant parser
        chunksize=chunksize
    )

    kept_parts = []

    for chunk in iterator:
        # Build mask depending on available keys
        masks = []

        if "SUBJECT_ID" in chunk.columns:
            masks.append(chunk["SUBJECT_ID"].isin(demo_subject_ids))

        if "HADM_ID" in chunk.columns:
            masks.append(chunk["HADM_ID"].isin(demo_hadm_ids))

        if demo_icustay_ids is not None and "ICUSTAY_ID" in chunk.columns:
            masks.append(chunk["ICUSTAY_ID"].isin(demo_icustay_ids))

        if not masks:   # table does not have any ID columns we know
            continue

        mask = masks[0]
        for m in masks[1:]:
            mask |= m

        filtered = chunk[mask]

        if not filtered.empty:
            kept_parts.append(filtered)

    if kept_parts:
        return pd.concat(kept_parts, ignore_index=True)
    else:
        return pd.DataFrame(columns=usecols)   # empty fallback

# %% [markdown]
# Enrich the demo dataset with notes and chart events.

# %%
# -------------------------------------------------------------
# Use enriched files if they already exist
# -------------------------------------------------------------

if NOTES_OUT.exists() and CHART_OUT.exists():
    print("Enriched files already exist, loading from disk.")

    demo_notes = pd.read_csv(NOTES_OUT)
    demo_chart = pd.read_csv(CHART_OUT)

# -------------------------------------------------------------
# Otherwise build them from scratch (slow step)
# -------------------------------------------------------------
else:
    print("Enriched files not found, running slow extraction...")

    # 1. Load demo tables
    demo_tables = {}
    for csv_path in DEMO_DIR.glob("*.csv"):
        name = csv_path.stem.upper()
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.upper()
        demo_tables[name] = df

    adm_demo = demo_tables["ADMISSIONS"]
    icu_demo = demo_tables["ICUSTAYS"]

    demo_subject_ids = set(adm_demo["SUBJECT_ID"])
    demo_hadm_ids    = set(adm_demo["HADM_ID"])
    demo_icustay_ids = set(icu_demo["ICUSTAY_ID"])

    # 2. Run extraction (extract_demo_rows defined earlier)
    demo_notes = extract_demo_rows(
        FULL_DIR,
        "NOTEEVENTS.csv.gz",
        demo_subject_ids=demo_subject_ids,
        demo_hadm_ids=demo_hadm_ids,
        usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
    )
    print("NOTEEVENTS extracted:", demo_notes.shape)

    demo_chart = extract_demo_rows(
        FULL_DIR,
        "CHARTEVENTS.csv.gz",
        demo_subject_ids=demo_subject_ids,
        demo_hadm_ids=demo_hadm_ids,
        demo_icustay_ids=demo_icustay_ids,
        usecols=[
            "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID",
            "ITEMID", "CHARTTIME", "VALUENUM",
        ],
    )
    print("CHARTEVENTS extracted:", demo_chart.shape)

    # 3. Save extracted tables
    demo_notes.to_csv(NOTES_OUT, index=False, compression="gzip")
    demo_chart.to_csv(CHART_OUT, index=False, compression="gzip")

    print("Extraction finished and saved.")

# %%
demo_notes.head()

# %%
demo_chart.head()

# %% [markdown]
# Aggregate notes data for individuals into separate csv files.

# %%
SUBJECT_NOTES_DIR = Path("subject_notes")
SUBJECT_NOTES_DIR.mkdir(exist_ok=True, parents=True)

for subject_id, df_sub in demo_notes.groupby("SUBJECT_ID"):
    # df_sub already contains HADM_ID, so stays are tracked implicitly
    out_path = SUBJECT_NOTES_DIR / f"subject_{subject_id}_notes.csv.gz"
    df_sub.to_csv(out_path, index=False, compression="gzip")

# %% [markdown]
# ### Load model

# %%


# %% [markdown]
# ### Inference

# %%



