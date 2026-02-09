# Research Resources Catalog

**Project:** "Do You Mean...?": Fixing User Intent Without Annoying Them
**Date:** 2026-02-08
**Hypothesis:** Autocomplete and intent detection systems often incorrectly correct users in ways that alter their original intent. Evaluating whether model rewrites preserve user intent is necessary.

---

## Papers (25 total)

### Query Correction & Spell Correction (5 papers)

| # | Title | Authors | Year | Venue | PDF |
|---|-------|---------|------|-------|-----|
| 1 | NeuSpell: A Neural Spelling Correction Toolkit | Jayanthi et al. | 2020 | EMNLP Demo | `papers/jayanthi2020_neuspell.pdf` |
| 2 | Correcting the Autocorrect: Context-Aware Typographic Error Correction | Zhang et al. | 2020 | arXiv | `papers/zhang2020_correcting_autocorrect.pdf` |
| 3 | Misspelling Correction with Pre-trained Language Models | Hu et al. | 2021 | arXiv | `papers/hu2021_misspelling_pretrained_lm.pdf` |
| 4 | Spelling Correction with Denoising Transformer | - | 2021 | arXiv | `papers/2021_spelling_denoising_transformer.pdf` |
| 5 | Multilingual Spell Checker for Production Search | Sharma et al. | 2023 | arXiv | `papers/sharma2023_multilingual_spellchecker.pdf` |

### Clarification Question Generation (6 papers)

| # | Title | Authors | Year | Venue | PDF |
|---|-------|---------|------|-------|-----|
| 6 | Interactive Question Clarification in Dialogue via RL | Hu et al. | 2020 | arXiv | `papers/2020_interactive_question_clarification_rl.pdf` |
| 7 | Resolving Intent Ambiguities by Retrieving Discriminative Clarifying Questions | Dhole | 2020 | arXiv | `papers/2020_resolving_intent_ambiguities.pdf` |
| 8 | Multi-Stage Clarification for Intent Detection | Lautraite | 2021 | arXiv | `papers/2021_multistage_clarification.pdf` |
| 9 | Zero-shot Clarifying Question Generation for Conversational Search | Wang et al. | 2023 | arXiv | `papers/2023_zeroshot_clarifying_questions.pdf` |
| 10 | Clarifying Ambiguities: Types and Approaches | - | 2025 | arXiv | `papers/2025_clarifying_ambiguities_types.pdf` |
| 11 | RAC: Retrieval-Augmented Clarification | Kebir et al. | 2026 | arXiv | `papers/2026_rac_retrieval_augmented_clarification.pdf` |

### Intent Detection & Classification (5 papers)

| # | Title | Authors | Year | Venue | PDF |
|---|-------|---------|------|-------|-----|
| 12 | Deep Search Query Intent Understanding | - | 2020 | arXiv | `papers/2020_deep_search_query_intent.pdf` |
| 13 | Conformal Intent Classification and Clarification (CICC) | den Hengst et al. | 2024 | arXiv | `papers/2024_conformal_intent_classification.pdf` |
| 14 | Intent Detection in the Age of LLMs | Arora et al. | 2024 | arXiv | `papers/2024_intent_detection_llms.pdf` |
| 15 | InteractComp: Benchmark for Ambiguous Query Understanding | Deng et al. | 2024 | arXiv | `papers/2024_interactcomp_ambiguous_queries.pdf` |
| 16 | Interpretability in Intent Detection | - | 2026 | arXiv | `papers/2026_interpretability_intent_detection.pdf` |

### Paraphrase & Semantic Evaluation (3 papers)

| # | Title | Authors | Year | Venue | PDF |
|---|-------|---------|------|-------|-----|
| 17 | On the Evaluation Metrics for Paraphrase Generation (ParaScore) | Shen et al. | 2022 | arXiv | `papers/2022_evaluation_metrics_paraphrase.pdf` |
| 18 | Paraphrasing, Entailment, and Similarity | - | 2022 | arXiv | `papers/2022_paraphrasing_entailment_similarity.pdf` |
| 19 | Effects of Paraphrasing on LLM Detection | - | 2024 | arXiv | `papers/2024_paraphrase_effects_llm_detection.pdf` |

### Query Rewriting & Reformulation (4 papers)

| # | Title | Authors | Year | Venue | PDF |
|---|-------|---------|------|-------|-----|
| 20 | Term-Based Query Reformulation | - | 2016 | arXiv | `papers/2016_term_based_query_reformulation.pdf` |
| 21 | Query Rewriting for RAG | - | 2023 | arXiv | `papers/2023_query_rewriting_rag.pdf` |
| 22 | Semantic Preservation in Text Rewriting | - | 2024 | arXiv | `papers/2024_rewrite_semantic_preservation.pdf` |
| 23 | Query Understanding with LLMs in Conversational Search | - | 2025 | arXiv | `papers/2025_query_understanding_llm_conversational.pdf` |

### Text Rewriting (2 papers)

| # | Title | Authors | Year | Venue | PDF |
|---|-------|---------|------|-------|-----|
| 24 | Sequential Decision Making for Inline Autocomplete | - | 2024 | arXiv | `papers/2024_sequential_decision_inline_autocomplete.pdf` |
| 25 | DR GENRE: RL from Decoupled LLM Feedback for Generic Text Rewriting | Li et al. | 2025 | arXiv | `papers/2025_dr_genre_rl_text_rewriting.pdf` |

---

## Datasets (14 cataloged, 7 downloaded)

### Downloaded (HIGH Priority)

| Dataset | Category | Size | Location |
|---------|----------|------|----------|
| BANKING77 | Intent Detection | 13,083 examples, 77 intents | `datasets/banking77/` |
| CLINC150 | Intent Detection | 23,700 examples, 150+OOS intents | `datasets/clinc150/` |
| PAWS | Semantic Similarity | 108,463 labeled pairs | `datasets/paws/` |
| STS-B | Semantic Similarity | 8,628 pairs, continuous 0-5 scores | `datasets/stsb/` |
| ClariQ | Clarification Questions | 18K single-turn + 1.8M multi-turn | `datasets/clariq/` |
| Qulac | Clarification Questions | 10K+ QA pairs, 198 topics | `datasets/qulac/` |

### Cataloged (MEDIUM Priority, Not Downloaded)

| Dataset | Category | How to Get |
|---------|----------|------------|
| GitHub Typo Corpus | Spell Correction | `git clone https://github.com/mhagiwara/github-typo-corpus.git` |
| NeuSpell/BEA-60K | Spell Correction | Via NeuSpell toolkit |
| Birkbeck Corpus | Spell Correction | https://www.dcs.bbk.ac.uk/~ROGER/corpora.html |
| SNIPS | Intent Detection | `load_dataset('DeepPavlov/snips')` |
| ClarQ | Clarification | `git clone https://github.com/vaibhav4595/ClarQ.git` |
| MRPC | Paraphrase | `load_dataset('glue', 'mrpc')` |
| AmazonQAC | Autocomplete | `load_dataset('amazon/AmazonQAC')` (395M samples) |
| ASSET | Text Rewriting | `load_dataset('asset', 'ratings')` |

Full catalog: `datasets/dataset_catalog.json`

---

## Code Repositories (8 cloned, 3 supplementary)

### Cloned Repositories

| Repo | Category | URL | Location |
|------|----------|-----|----------|
| NeuSpell | Spell Correction | https://github.com/neuspell/neuspell | `code/neuspell/` |
| Few-Shot-Intent-Detection | Intent Detection | https://github.com/jianguoz/Few-Shot-Intent-Detection | `code/Few-Shot-Intent-Detection/` |
| BERTScore | Evaluation Metric | https://github.com/Tiiiger/bert_score | `code/bert_score/` |
| ParaScore | Evaluation Metric | https://github.com/shadowkiller33/ParaScore | `code/ParaScore/` |
| BARTScore | Evaluation Metric | https://github.com/neulab/BARTScore | `code/BARTScore/` |
| Conformal Prediction | Uncertainty | https://github.com/aangelopoulos/conformal-prediction | `code/conformal-prediction/` |
| InfoCQR | Query Rewriting | https://github.com/smartyfh/InfoCQR | `code/InfoCQR/` |
| Sentence-Transformers | Embeddings | https://github.com/huggingface/sentence-transformers | `code/sentence-transformers/` |

### Supplementary (Not Cloned)

| Repo | URL | Description |
|------|-----|-------------|
| ACQSurvey | https://github.com/rahmanidashti/ACQSurvey | Survey of clarification question datasets |
| Conformal NLP | https://github.com/PatrikDurdevic/Conformal-Prediction-for-NLP-Sentiment-Classification | Conformal prediction for NLP |
| CAsT BART | https://github.com/carlos-gemmell/CAsT_BART_query_rewriting | BART query rewriting |

Full survey: `artifacts/github_repository_survey.json`

---

## Pre-trained Models

| Model | Source | Use Case |
|-------|--------|----------|
| BERT spell correction | `murali1996/bert-base-cased-spell-correction` (HuggingFace) | Baseline spelling correction |
| DeBERTa-xlarge-mnli | `microsoft/deberta-xlarge-mnli` (HuggingFace) | Best BERTScore correlation |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) | Fast semantic similarity |
| all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2` (HuggingFace) | High-quality embeddings |
| BART-large | `facebook/bart-large` (HuggingFace) | BARTScore evaluation |

---

## Key Metrics for Evaluation

| Metric | Type | Library | Best For |
|--------|------|---------|----------|
| BERTScore (ref-free) | Semantic similarity | `bert_score` | Primary intent preservation signal |
| ParaScore | Semantic + divergence | `ParaScore` | Combined quality evaluation |
| BARTScore | Generative evaluation | `BARTScore` | Informativeness assessment |
| Edit Ratio | Word-level distance | Custom | Measuring unnecessary changes |
| Intent Label Shift | Classification change | Intent classifiers | Binary intent violation detection |
| NLI Score | Entailment | HuggingFace models | Bidirectional meaning preservation |

---

## Directory Structure

```
fix-user-intent-nlp-claude/
├── literature_review.md          # Comprehensive literature synthesis
├── resources.md                  # This file - complete resource catalog
├── .resource_finder_complete     # Completion marker
├── papers/
│   ├── README.md                 # Paper listing with metadata
│   ├── *.pdf                     # 25 downloaded papers
│   └── pages/                    # Chunked papers for reading
├── datasets/
│   ├── README.md                 # Dataset documentation
│   ├── .gitignore                # Excludes data files
│   ├── dataset_catalog.json      # Structured catalog (14 datasets)
│   ├── download_datasets.py      # Download helper script
│   ├── banking77/                # BANKING77 intent detection
│   ├── clinc150/                 # CLINC150 intent detection
│   ├── paws/                     # PAWS paraphrase adversaries
│   ├── stsb/                     # STS-B semantic similarity
│   ├── clariq/                   # ClariQ clarification questions
│   └── qulac/                    # Qulac clarification questions
├── code/
│   ├── README.md                 # Repository documentation
│   ├── neuspell/                 # Spelling correction toolkit
│   ├── Few-Shot-Intent-Detection/ # Intent detection benchmarks
│   ├── bert_score/               # BERTScore metric
│   ├── ParaScore/                # ParaScore metric
│   ├── BARTScore/                # BARTScore metric
│   ├── conformal-prediction/     # Conformal prediction framework
│   ├── InfoCQR/                  # Query rewriting
│   └── sentence-transformers/    # Text embeddings
├── artifacts/
│   └── github_repository_survey.json  # Structured repo survey
└── paper_search_results/         # Raw search results
```
