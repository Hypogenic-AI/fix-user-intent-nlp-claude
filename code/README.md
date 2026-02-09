# Code Repositories

**Project:** "Do You Mean...?": Fixing User Intent Without Annoying Them

This directory contains cloned code repositories relevant to the research project.

## Repository Catalog

See `../artifacts/github_repository_survey.json` for the full structured survey with relevance annotations.

## Cloned Repositories

### Spell Correction
| Repo | Description | Key Use |
|------|-------------|---------|
| `neuspell/` | [NeuSpell](https://github.com/neuspell/neuspell) - Neural spelling correction toolkit | Baseline spell correction models; pretrained BERT-based corrector |

### Intent Detection
| Repo | Description | Key Use |
|------|-------------|---------|
| `Few-Shot-Intent-Detection/` | [Few-Shot-Intent-Detection](https://github.com/jianguoz/Few-Shot-Intent-Detection) | BANKING77, CLINC150, HWU64, SNIPS, ATIS datasets with baselines |

### Evaluation Metrics
| Repo | Description | Key Use |
|------|-------------|---------|
| `bert_score/` | [BERTScore](https://github.com/Tiiiger/bert_score) | Semantic similarity for intent preservation measurement |
| `ParaScore/` | [ParaScore](https://github.com/shadowkiller33/ParaScore) | Paraphrase evaluation with divergence modeling |
| `BARTScore/` | [BARTScore](https://github.com/neulab/BARTScore) | Generative text evaluation metric |
| `sentence-transformers/` | [Sentence-Transformers](https://github.com/huggingface/sentence-transformers) | Text embeddings for semantic similarity |

### Uncertainty & Conformal Prediction
| Repo | Description | Key Use |
|------|-------------|---------|
| `conformal-prediction/` | [Conformal Prediction](https://github.com/aangelopoulos/conformal-prediction) | Coverage-guaranteed prediction sets for intent classification |

### Query Rewriting
| Repo | Description | Key Use |
|------|-------------|---------|
| `InfoCQR/` | [InfoCQR](https://github.com/smartyfh/InfoCQR) | Conversational query rewriting with intent preservation |

## Additional Repositories (Not Cloned)

These repositories were identified as relevant but not cloned. Clone as needed:

| Repo | URL | Description |
|------|-----|-------------|
| ACQSurvey | https://github.com/rahmanidashti/ACQSurvey | Survey of clarification question datasets |
| Conformal NLP Classification | https://github.com/PatrikDurdevic/Conformal-Prediction-for-NLP-Sentiment-Classification | Conformal prediction for NLP |
| CAsT BART Query Rewriting | https://github.com/carlos-gemmell/CAsT_BART_query_rewriting | BART-based query rewriting |

## Usage

All repositories are shallow clones (`--depth 1`) to save disk space. To get full history:

```bash
cd <repo_directory>
git fetch --unshallow
```
