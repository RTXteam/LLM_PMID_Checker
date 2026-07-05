# LLM TMKP Checker

A system for checking whether research triples are supported by TMKP (Text Mining Knowledge Provider) provided text snippets, using large language models.

## Overview

Given a Parquet (or TSV) file of research triples (e.g., `aspirin biolink:treats_or_applied_or_studied_to_treat headache`) with TMKP supporting text, this system:

1. **Evaluates support** using vLLM-served models with concurrent batch processing
2. **Saves results** to a SQLite database (recommended) or TSV file, preserving all input columns alongside evaluation outputs

## Pre-computed Evaluation Results

Pre-computed evaluation results for the TMKP KGX dataset are available as a [GitHub release](https://github.com/RTXteam/LLM_PMID_Checker/releases/tag/tmkp-v1.0). TMKP provides inline supporting text extracted from publications, so evaluation is performed directly against those text snippets.

To download the pre-computed results, please use the provided download script (requires only Python stdlib):

```bash
python scripts/download_release_data.py --output-dir results --tag tmkp-v1.0
```

After running the command above, the following file will be downloaded:
```
results/
    results.parquet                              <-- use this (557 MB, 3,199,267 rows with 17 cols)
    TMKP_Sentences_Evaluation_v1.0.tar.gz
```

#### Explanation of Results

The release contains a Parquet file:

| File | Size | Rows | Description |
|---|---|---|---|
| `results.parquet` | ~557 MB | 3,199,267 | LLM evaluation of TMKP triples against their supporting text |

**`results.parquet`** columns:

| Column | Type | Description |
|---|---|---|
| `subject_curie` | string | Subject entity CURIE |
| `subject_name` | string | Subject entity name |
| `predicate` | string | Biolink predicate |
| `object_curie` | string | Object entity CURIE |
| `object_name` | string | Object entity name |
| `supporting_text` | string | Text snippet from the publication |
| `supporting_text_id` | string | Unique identifier for the supporting text |
| `subject_location_in_text` | string | Character offsets of subject in text |
| `object_location_in_text` | string | Character offsets of object in text |
| `extraction_confidence_score` | string | Original TMKP extraction confidence |
| `supporting_document_year` | string | Publication year |
| `supporting_text_section_type` | string | Section type (e.g., ABSTRACT, RESULTS, INTRO) |
| `predicted` | string | Whether the triple is supported (`True`/`False`) |
| `support` | string | LLM judgment: `yes`, `no`, or `maybe` |
| `subject_mentioned` | string | Whether the subject is mentioned in text |
| `object_mentioned` | string | Whether the object is mentioned in text |
| `reasoning` | string | LLM's reasoning for the judgment |

Each row represents one unique `(subject_curie, predicate, object_curie, supporting_text_id)` combination. The LLM reads the supporting text, checks whether the subject and object are mentioned, and judges whether the text supports the stated relationship.

#### TMKP Dataset Statistics

**Coverage:**

| Metric | Count |
|---|---|
| Total evaluated triples | 3,199,267 |
| Unique subject CURIEs | 23,527 |
| Unique object CURIEs | 22,153 |
| Unique Biolink predicates | 3 |

**Predicate distribution:**

| Predicate | Count | Percentage |
|---|---|---|
| `biolink:affects` | 2,091,282 | 65.4% |
| `biolink:treats_or_applied_or_studied_to_treat` | 873,268 | 27.3% |
| `biolink:contributes_to` | 234,717 | 7.3% |

**Support distribution:**

| Support | Count | Percentage |
|---|---|---|
| `yes` | 1,755,096 | 54.9% |
| `no` | 1,228,264 | 38.4% |
| `maybe` | 215,907 | 6.7% |

**Entity mention rates:**

| Metric | Count | Percentage |
|---|---|---|
| Both subject and object mentioned | 2,436,838 | 76.2% |
| Subject only mentioned | 486,835 | 15.2% |
| Object only mentioned | 194,861 | 6.1% |
| Neither mentioned | 80,733 | 2.5% |

## How to Run

### 1. Install Dependencies

```bash
conda activate llm_pmid_env
pip install -r requirements.txt
```

### 2. Prepare TMKP Data

Download the unchanged TMKP source data from [Translator KGX STORAGE](https://kgx-storage.ci.transltr.io/data/tmkp/tmkp-2024-09-07/source_data/) into `data/tmkp_kgx/`, then extract edges from :

```bash
python scripts/extract_tmkp_edges.py \
    -i data/tmkp_kgx/normalized_edges.jsonl \
    -o data/tmkp_kgx/tmkp_edges_extracted.parquet
```

### 3. Node File & CURIE Names for Richer Entity Context (Recommended)

Providing entity metadata improves evaluation quality by giving the LLM richer context about subject/object names:

```bash
python scripts/extract_curie_names.py \
    --input data/tmkp_kgx/tmkp_edges_extracted.parquet \
    --output data/tmkp_kgx/curie_all_names.tsv \
    --batch-size 500 --max-concurrent 10
```

This queries the [Node Normalization API](https://nodenormalization-sri.renci.org/docs) for all unique CURIEs and collects every known name variant (primary label + labels from equivalent identifiers), case-insensitively deduplicated.

### 4. Start vLLM Server(s)

Use the provided setup script to launch one or more vLLM servers:

```bash
# GPT-OSS 120B on GPU 0
VLLM_MODEL=openai/gpt-oss-120b VLLM_MODEL_NAME=gpt-oss-120b-vllm VLLM_GPU=0 VLLM_PORT=8000 bash setup_vllm.sh
```

### 5. Extract Biolink Predicate Definitions

Extract predicate definitions to provide the LLM with formal predicate semantics:

```bash
python scripts/extract_biolink_predicates.py \
    --input data/biolink_data/biolink-model.yaml \
    --output data/biolink_data/biolink_predicates.tsv
```

### 6. Configure Environment

Create a `.env` file in the project root:

```bash
# Batch processing
MAX_CONCURRENT_REQUESTS=24

# vLLM Configuration
VLLM_BASE_URL=http://localhost:8000

# Per-model URLs (comma-separated model=url pairs)
VLLM_MODEL_URLS=gpt-oss-20b-vllm=http://localhost:8000,gpt-oss-120b-vllm=http://localhost:8002

# Available vLLM models (must match --served-model-name used when starting vLLM)
AVAILABLE_VLLM_MODELS=gpt-oss-20b-vllm,gpt-oss-120b-vllm
```

### 7. Run Evaluation

See [Usage](#usage) below for full command-line options and examples.

## Usage

```
python main.py --input INPUT_FILE --output OUTPUT_FILE [options]
```

Input format is auto-detected from the file extension:
- `.parquet` / `.pq` → Parquet (recommended, preserves text exactly)
- `.tsv` / `.txt` → Tab-separated values

Output format is auto-detected from the file extension:
- `.db` / `.sqlite` / `.sqlite3` → SQLite database (recommended for stop/resume)
- `.tsv` / `.txt` → Tab-separated values

| Flag | Description |
|---|---|
| `--input` | **(required)** Input file (`.parquet` or `.tsv`; must contain `subject_curie`, `predicate`, `object_curie`, `supporting_text`) |
| `--output` | **(required)** Output file (`.db` for SQLite recommended, `.tsv` for TSV) |
| `--val_model` | Validation model (default: first in `AVAILABLE_VLLM_MODELS`) |
| `--round2_model` | Optional Round 2 model for re-evaluating yes/maybe results |
| `--table` | SQLite table name, only for `.db` output (default: `evaluations`) |
| `--node_dict` | Nodes file (`.jsonl`, `.jsonl.gz`) for richer entity context |
| `--names_file` | `curie_all_names.tsv` to supplement `--node_dict` with richer equivalent names |
| `--predicate_file` | Biolink predicates TSV with predicate definitions (columns: `predicate`, `description`) |
| `--max_concurrent` | Max concurrent requests (default: `MAX_CONCURRENT_REQUESTS` from `.env`) |
| `--overwrite` | Discard existing output and start fresh (default: auto-resume) |
| `--verbose` / `-v` | Enable DEBUG logging |

### Stop & Resume

Results are written incrementally -- every completed row is flushed to disk immediately. You can safely `Ctrl+C` at any time and re-run the exact same command to resume:

```bash
# First run (or resume after interruption) -- same command each time
python main.py --input data/tmkp_kgx/tmkp_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-120b-vllm --max_concurrent 24 \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/tmkp_kgx/nodes.jsonl \
    --names_file data/tmkp_kgx/curie_all_names.tsv
```

On resume, the program reads the existing output, determines which `(subject_curie, predicate, object_curie, supporting_text_id)` rows are already evaluated, and only processes the remaining rows.

### Examples

```bash
# Standard TMKP evaluation with Parquet input and SQLite output
python main.py --input data/tmkp_kgx/tmkp_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-120b-vllm --max_concurrent 24 \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/tmkp_kgx/nodes.jsonl \
    --names_file data/tmkp_kgx/curie_all_names.tsv

# Two-round evaluation (Round 1 with 20B, Round 2 with 120B)
python main.py --input data/tmkp_kgx/tmkp_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-20b-vllm --round2_model gpt-oss-120b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv
```

## Input Format

The input file (Parquet or TSV) **must** contain these columns:

| Column | Description |
|---|---|
| `subject_curie` | Subject entity CURIE (e.g., `CHEBI:70723`) |
| `predicate` | Relationship (e.g., `biolink:affects`, `biolink:treats_or_applied_or_studied_to_treat`) |
| `object_curie` | Object entity CURIE (e.g., `PR:000004517`) |
| `supporting_text` | Text snippet to evaluate the triple against |

Any additional columns are carried through to the output unchanged.

## Output Format

Results are written to a **SQLite database** (`.db`, recommended) or a **TSV file** (`.tsv`), depending on the `--output` extension. Both formats contain all columns from the input plus these evaluation columns:

| Column | Type | Description |
|---|---|---|
| `predicted` | bool | Whether the triple is supported (`support == "yes"`) |
| `support` | text | `yes`, `no`, or `maybe` |
| `subject_mentioned` | bool | Whether the subject appears in the supporting text |
| `object_mentioned` | bool | Whether the object appears in the supporting text |
| `reasoning` | text | LLM's reasoning for the judgment |

### Post-evaluation Utilities

**Convert SQLite to Parquet** (for final delivery or analytical queries):

```bash
python scripts/convert_db_to_parquet.py --db results.db --output-dir .
```

**Verify coverage** (ensure all input rows are accounted for):

```bash
python scripts/compare_coverage.py \
    --extracted data/tmkp_kgx/tmkp_edges_extracted.parquet \
    --results-db results.db
```

## Available Models

| Model | HuggingFace Repo |
|---|---|
| `gpt-oss-20b-vllm` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| `gpt-oss-120b-vllm` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
