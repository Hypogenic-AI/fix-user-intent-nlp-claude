# Papers Collection

**Project:** "Do You Mean...?": Fixing User Intent Without Annoying Them

This directory contains 25 research papers collected for this project, covering autocomplete/intent correction, clarification question generation, intent detection, paraphrase evaluation, and text rewriting.

## Papers by Category

### Query Correction & Spell Correction
| File | Title | Year | Key Contribution |
|------|-------|------|-----------------|
| `jayanthi2020_neuspell.pdf` | NeuSpell: A Neural Spelling Correction Toolkit | 2020 | Open-source toolkit with 10 spell checkers and BEA-60K benchmark |
| `zhang2020_correcting_autocorrect.pdf` | Correcting the Autocorrect: Context-Aware Typographic Error Correction | 2020 | Twitter Typo Corpus; context-aware correction via visual similarity + phonetics |
| `hu2021_misspelling_pretrained_lm.pdf` | Misspelling Correction with Pre-trained Language Models | 2021 | BERT-based correction preserving context |
| `2021_spelling_denoising_transformer.pdf` | Spelling Correction with Denoising Transformer | 2021 | Denoising approach to spell correction |
| `sharma2023_multilingual_spellchecker.pdf` | Multilingual Spell Checker for Production Search | 2023 | Production-scale spell correction for search queries |

### Clarification Question Generation
| File | Title | Year | Key Contribution |
|------|-------|------|-----------------|
| `2020_interactive_question_clarification_rl.pdf` | Interactive Question Clarification in Dialogue via RL | 2020 | MCTS-based RL for clarification, 66.36% CTR in production |
| `2020_resolving_intent_ambiguities.pdf` | Resolving Intent Ambiguities by Retrieving Discriminative Clarifying Questions | 2020 | QG + template fallback for discriminative questions |
| `2021_multistage_clarification.pdf` | Multi-Stage Clarification for Intent Detection | 2021 | Confidence threshold pipeline (75%), CLINC150/SCOPE evaluation |
| `2023_zeroshot_clarifying_questions.pdf` | Zero-shot Clarifying Question Generation for Conversational Search | 2023 | Zero-shot beats supervised baselines, NeuroLogic Decoding |
| `2025_clarifying_ambiguities_types.pdf` | Clarifying Ambiguities: Types and Approaches | 2025 | Taxonomy of ambiguity types requiring clarification |
| `2026_rac_retrieval_augmented_clarification.pdf` | RAC: Retrieval-Augmented Clarification with DPO | 2026 | DPO-trained faithful clarifying question generation |

### Intent Detection & Classification
| File | Title | Year | Key Contribution |
|------|-------|------|-----------------|
| `2020_deep_search_query_intent.pdf` | Deep Search Query Intent Understanding | 2020 | Deep learning for search query intent |
| `2024_conformal_intent_classification.pdf` | Conformal Intent Classification and Clarification (CICC) | 2024 | Conformal prediction sets with coverage guarantees for intent classification |
| `2024_intent_detection_llms.pdf` | Intent Detection in the Age of LLMs | 2024 | LLMs outperform SetFit by ~8%, hybrid routing for efficiency |
| `2024_interactcomp_ambiguous_queries.pdf` | InteractComp: Benchmark for Ambiguous Query Understanding | 2024 | 210 questions, models achieve only 13.73% on ambiguous queries |
| `2026_interpretability_intent_detection.pdf` | Interpretability in Intent Detection | 2026 | Explainable intent detection methods |

### Paraphrase & Semantic Evaluation
| File | Title | Year | Key Contribution |
|------|-------|------|-----------------|
| `2022_evaluation_metrics_paraphrase.pdf` | On the Evaluation Metrics for Paraphrase Generation (ParaScore) | 2022 | ParaScore: reference-free > reference-based; sectional divergence function |
| `2022_paraphrasing_entailment_similarity.pdf` | Paraphrasing, Entailment, and Similarity | 2022 | Cross-metric evaluation of semantic preservation |
| `2024_paraphrase_effects_llm_detection.pdf` | Effects of Paraphrasing on LLM Detection | 2024 | How paraphrasing affects detectability |

### Query Rewriting & Reformulation
| File | Title | Year | Key Contribution |
|------|-------|------|-----------------|
| `2016_term_based_query_reformulation.pdf` | Term-Based Query Reformulation | 2016 | Foundation for query reformulation techniques |
| `2023_query_rewriting_rag.pdf` | Query Rewriting for RAG | 2023 | LLM-based query rewriting for retrieval |
| `2024_rewrite_semantic_preservation.pdf` | Semantic Preservation in Text Rewriting | 2024 | Measuring meaning preservation in rewrites |
| `2025_query_understanding_llm_conversational.pdf` | Query Understanding with LLMs in Conversational Search | 2025 | LLM-based conversational query understanding |

### Text Rewriting with Intent Preservation
| File | Title | Year | Key Contribution |
|------|-------|------|-----------------|
| `2024_sequential_decision_inline_autocomplete.pdf` | Sequential Decision Making for Inline Autocomplete | 2024 | MDP framework for autocomplete suggestions |
| `2025_dr_genre_rl_text_rewriting.pdf` | DR GENRE: RL from Decoupled LLM Feedback for Generic Text Rewriting | 2025 | Decoupled rewards (agreement, coherence, conciseness) for rewriting |

## Chunked Papers

The `pages/` subdirectory contains papers split into 3-page chunks for detailed reading. Each chunk file follows the naming pattern: `{paper_name}_chunk_{NNN}.pdf`.

## How to Add Papers

1. Download the PDF to this directory
2. Name it descriptively: `{year}_{short_description}.pdf`
3. Update this README with the paper details
