# Datasets

**Project:** "Do You Mean...?": Fixing User Intent Without Annoying Them

This directory contains datasets for evaluating intent preservation in autocomplete/correction systems.

## Dataset Catalog

See `dataset_catalog.json` for the full structured catalog with download commands, licenses, and relevance annotations.

## HIGH Priority Datasets

### Intent Detection Benchmarks

| Dataset | Size | Intents | Download |
|---------|------|---------|----------|
| **BANKING77** | 13,083 examples | 77 banking intents | `load_dataset('PolyAI/banking77')` |
| **CLINC150** | 23,700 examples | 150 intents + OOS | `load_dataset('clinc_oos', 'plus')` |
| **SNIPS** | ~14,484 examples | 7 intents + slots | `load_dataset('DeepPavlov/snips')` |

### Clarification Question Datasets

| Dataset | Size | Format | Download |
|---------|------|--------|----------|
| **ClariQ** | 18K single-turn + 1.8M multi-turn | TSV | `git clone https://github.com/aliannejadi/ClariQ.git` |
| **Qulac** | 10K+ QA pairs, 198 topics | JSON | `git clone https://github.com/aliannejadi/qulac.git` |

### Semantic Similarity / Paraphrase

| Dataset | Size | Score Type | Download |
|---------|------|------------|----------|
| **PAWS** | 108K labeled pairs | Binary (paraphrase/not) | `load_dataset('google-research-datasets/paws', 'labeled_final')` |
| **STS-B** | 8,628 pairs | Continuous 0-5 | `load_dataset('sentence-transformers/stsb')` |

### Query Autocomplete

| Dataset | Size | Format | Download |
|---------|------|--------|----------|
| **AmazonQAC** | 395M train, 20K test | Parquet | `load_dataset('amazon/AmazonQAC')` |

## MEDIUM Priority Datasets

| Dataset | Size | Category | Download |
|---------|------|----------|----------|
| **GitHub Typo Corpus** | 350K+ edits | Spell Correction | `git clone https://github.com/mhagiwara/github-typo-corpus.git` |
| **NeuSpell/BEA-60K** | 60K pairs | Spell Correction | Via NeuSpell toolkit |
| **Birkbeck Corpus** | 36K misspellings | Spell Correction | https://www.dcs.bbk.ac.uk/~ROGER/corpora.html |
| **ClarQ** | ~2M examples | Clarification | `git clone https://github.com/vaibhav4595/ClarQ.git` |
| **MRPC** | 5,801 pairs | Paraphrase | `load_dataset('glue', 'mrpc')` |
| **ASSET** | 23,590 pairs | Text Rewriting | `load_dataset('asset', 'ratings')` |

## Download Helper

Use the included download script for batch downloading:

```bash
# List all datasets
python download_datasets.py --list

# Download all HIGH priority datasets
python download_datasets.py --priority HIGH

# Download specific datasets by ID
python download_datasets.py --ids 7 8 10 12

# Download all datasets
python download_datasets.py --all
```

## Directory Structure

```
datasets/
├── README.md                  # This file
├── .gitignore                 # Excludes data files from git
├── dataset_catalog.json       # Full structured catalog
├── download_datasets.py       # Download helper script
├── banking77/                 # BANKING77 dataset (downloaded)
├── clinc150/                  # CLINC150 dataset (downloaded)
├── paws/                      # PAWS dataset (downloaded)
├── stsb/                      # STS-B dataset (downloaded)
├── clariq/                    # ClariQ dataset (cloned)
└── qulac/                     # Qulac dataset (cloned)
```

## Notes

- Large datasets (AmazonQAC: 395M samples) are not pre-downloaded due to size; use the download commands above.
- The `.gitignore` excludes all downloaded data files. Only the catalog, scripts, and README are tracked in git.
- All datasets require the `datasets` Python package for HuggingFace downloads: `pip install datasets`
