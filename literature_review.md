# Literature Review: "Do You Mean...?": Fixing User Intent Without Annoying Them

## Research Hypothesis

Autocomplete and intent detection systems often incorrectly correct users in ways that alter their original intent. Evaluating whether model rewrites preserve user intent is necessary.

## 1. Problem Definition & Motivation

The central tension in autocomplete and query correction systems is between **helpfulness** (correcting genuine errors) and **respecting user intent** (not changing what the user actually meant). When a user types a query with a typo, misspelling, or ambiguous phrasing, the system must decide: should it silently correct, ask for clarification, or leave the query as-is?

This literature review synthesizes 25 papers across five research areas that collectively address this problem: (1) spelling/query correction, (2) clarification question generation, (3) intent detection and classification, (4) semantic preservation metrics, and (5) text rewriting with intent preservation.

---

## 2. Spelling & Query Correction

### 2.1 Neural Spelling Correction

**NeuSpell** (Jayanthi et al., 2020) provides an open-source toolkit comprising 10 neural spell checkers evaluated on naturally occurring misspellings (BEA-60K benchmark). The key insight for our work is that neural correctors can be context-sensitive, using surrounding words to disambiguate corrections. However, even the best models make errors when domain-specific terminology or proper nouns are involved -- precisely the cases where "correcting" alters user intent.

**Zhang et al. (2020)** address this directly in "Correcting the Autocorrect," introducing the **Twitter Typo Corpus** of naturally occurring typos. Their approach uses visual similarity and phonetic features alongside contextual signals. The Twitter Typo Corpus is valuable because it captures real user behavior where autocorrect frequently "fixes" intentional non-standard spellings (slang, abbreviations, proper nouns).

**Sharma et al. (2023)** describe a production-scale multilingual spell checker for search queries, highlighting the practical challenges: spell correction must handle code-switching, transliteration, and domain jargon. In production, aggressive correction rates lead to user frustration when the system overrides intentional queries.

### 2.2 Key Takeaway for Our Project

Spelling correction is the most common form of "fixing" user input, and also the most common source of intent violation. The literature shows a clear gap: existing systems optimize for correction accuracy (was the typo fixed?) rather than intent preservation (did the correction maintain what the user wanted?). Our project can fill this gap by framing spelling correction as an intent-preservation problem.

---

## 3. Clarification Question Generation

### 3.1 When to Ask vs. When to Correct

The clarification question literature addresses a fundamental design choice: when should a system ask "Do you mean...?" versus silently correcting?

**Hu et al. (2020)** present an RL-based approach using Monte Carlo Tree Search (MCTS) for selecting clarification questions in task-oriented dialogue. Their system was deployed in production, achieving **66.36% CTR** (click-through rate on clarification suggestions). The key finding is that clarification is most effective when: (a) the system has moderate confidence, (b) the cost of misinterpretation is high, and (c) the user has shown willingness to engage. Their label-based clarification approach (presenting options rather than open-ended questions) is particularly relevant -- it bounds the cognitive load on users.

**Dhole (2020)** tackles discriminative clarifying questions for intent disambiguation. The approach combines question generation with template-based fallbacks, finding that pure QG achieves only **34% coverage** -- meaning in 66% of ambiguous cases, the generated question fails to discriminate between candidate intents. This highlights a practical limitation: generating good clarifying questions is hard, and bad ones are worse than no question at all.

**Lautraite (2021)** proposes the most directly relevant system: a multi-stage clarification pipeline with explicit confidence thresholds. The system operates in three stages:
1. **High confidence (>75%)**: Auto-correct without asking
2. **Moderate confidence**: Present clarification question
3. **Low confidence**: Escalate or ask open-ended question

This threshold-based approach is exactly the kind of decision framework our project needs. The 75% threshold was empirically determined on CLINC150/SCOPE datasets.

### 3.2 Zero-Shot and Recent Approaches

**Wang et al. (2023)** show that zero-shot clarifying question generation (using constrained decoding via NeuroLogic Decoding) **outperforms supervised baselines** on naturalness (82.6% rated "Good"). This is significant because it means we can generate clarifying questions without task-specific training data, making the approach more generalizable.

**Kebir et al. (2026)** introduce **RAC** (Retrieval-Augmented Clarification), which uses DPO (Direct Preference Optimization) to train models that generate faithful clarifying questions grounded in retrieved evidence. This addresses the hallucination problem in clarification -- asking about distinctions that don't actually exist.

### 3.3 Key Takeaway for Our Project

The clarification literature provides strong evidence that a confidence-threshold approach is effective: correct when confident, clarify when uncertain, escalate when very uncertain. The challenge is calibrating confidence appropriately and generating clarification questions that users find helpful rather than annoying. The 66.36% CTR from Hu et al. suggests that well-designed clarification is acceptable to users.

---

## 4. Intent Detection & Classification

### 4.1 Modern Intent Detection

**Arora et al. (2024)** benchmark LLMs against traditional intent detection methods, finding that LLMs outperform SetFit by approximately **8%** on standard benchmarks. However, they also show that a **hybrid routing approach** (using a smaller model for confident predictions and routing uncertain cases to an LLM) achieves comparable accuracy with much lower latency. This hybrid approach maps well to our correction-vs-clarification decision.

### 4.2 Conformal Prediction for Intent Classification

**Den Hengst et al. (2024)** present **CICC** (Conformal Intent Classification and Clarification), the paper most directly aligned with our project's thesis. Their approach uses conformal prediction to produce **prediction sets with statistical coverage guarantees**:

- Given a desired coverage level (e.g., 1-α = 0.95), the system produces a set of possible intents guaranteed to contain the true intent with probability ≥ 95%.
- If the prediction set contains a single intent → answer directly
- If the prediction set contains 2-7 intents → generate a clarifying question listing the candidates
- If the prediction set contains >7 intents → the query is too ambiguous; escalate

Key results across 8 benchmarks (ACID, ATIS, B77, C150, HWU64, IND, MTOD):
- CICC achieves target coverage while maintaining **highest single-answer rate** compared to baselines
- On B77 with optimized α: 97% coverage, 92% single-answer rate, average CQ size of 2.32
- Clarifying questions are generated using Vicuna-7B with few-shot prompting

The cognitive load threshold of **7 options** (from Miller's 1956 "magical number seven") provides a principled upper bound on clarification complexity.

### 4.3 Handling Ambiguous Queries

**Deng et al. (2025)** introduce **InteractComp**, a benchmark for search agent ambiguity recognition. Their key finding: models achieve only **13.73%** accuracy on ambiguous queries compared to **71.5%** when given contextual clarification. This dramatic gap quantifies the cost of NOT asking clarifying questions when queries are ambiguous.

### 4.4 Key Takeaway for Our Project

Conformal prediction provides the most principled framework for our problem. It gives statistical guarantees on coverage (the true intent is in the prediction set) while minimizing unnecessary clarification (keeping prediction sets small). The CICC framework directly implements the "correct when confident, clarify when uncertain" paradigm with rigorous statistical foundations.

---

## 5. Semantic Preservation Metrics

### 5.1 Measuring Intent Preservation

A critical component of our project is measuring whether a rewrite/correction preserves the original intent. Several metrics have been proposed:

**BERTScore** (Zhang et al., 2019): Computes token-level similarity using contextual BERT embeddings. The recommended model (deberta-xlarge-mnli) achieves highest correlation with human judgments. Key advantage: robust to paraphrases and word-order changes.

**BARTScore** (Yuan et al., 2021): Evaluates text as a generation problem using BART's log-likelihood. Can assess informativeness, fluency, and factuality independently. Outperforms other metrics in 16/22 evaluation settings.

**ParaScore** (Shen et al., 2022): The most relevant metric for our work. Key findings:
1. **Reference-free metrics outperform reference-based metrics** for paraphrase evaluation. This is critical because we typically don't have a "gold reference" correction.
2. **Lexical divergence matters, but only up to a threshold** (γ=0.35 NED). Beyond that threshold, more divergence doesn't improve quality. This maps perfectly to our setting: small corrections (typo fixes) should have low divergence and are easier to evaluate.
3. The ParaScore formula: `ParaScore = max(Sim(X,C), Sim(R,C)) + ω·DS(X,C)`, where DS is a sectional function that rewards divergence up to the threshold.

On controlled experiments:
- BERTScore achieves **0.785 Pearson correlation** with human judgments on semantic-preservation subsets
- ParaScore achieves **0.522 Pearson** overall (vs. 0.491 for best existing metric)
- On extended datasets with trivial copies: ParaScore **0.527** vs. BERTScore **0.316**

### 5.2 The Edit Distance Insight

DR GENRE (Li et al., 2025) introduces **edit ratio** as a conciseness metric: the relative word-level edit distance between original and rewritten text. Lower edit ratio means fewer unnecessary changes. This directly operationalizes the "don't annoy the user" part of our hypothesis -- excessive, unsolicited edits are annoying.

### 5.3 Key Takeaway for Our Project

For measuring intent preservation, we should use:
1. **BERTScore (reference-free mode)** as the primary semantic preservation signal
2. **ParaScore** for combined semantic + divergence evaluation
3. **Edit ratio** to penalize excessive/unnecessary changes
4. **Intent label shift** (using intent classifiers on BANKING77/CLINC150) as a proxy for intent violation

Avoid BLEU and ROUGE -- they show poor correlation with human judgments of paraphrase quality.

---

## 6. Text Rewriting with Intent Preservation

### 6.1 Decoupled Reward Optimization

**DR GENRE** (Li et al., 2025) is the most sophisticated framework for intent-preserving text rewriting. The key innovation is **decoupled, task-weighted rewards** during RL fine-tuning:

1. **Agreement reward**: Did the rewrite follow the instruction? (the "fix" part)
2. **Coherence reward**: Is the result internally consistent? (broken coherence = annoying)
3. **Conciseness reward (edit ratio)**: Were unnecessary parts left unchanged? (excessive editing = annoying)

These three rewards map directly to our "fix without annoying" paradigm:
- Agreement = making the necessary correction
- Coherence = not breaking the surrounding context
- Conciseness = not changing what doesn't need changing

Task-specific weighting is critical: conversational rewrites need high agreement weight (9/16), while factual corrections need high coherence weight (6/16). The system uses PPO (not DPO) because PPO allows exploration beyond the initial policy.

Results on three benchmarks:
- Factuality: F1@13 = 0.8091, coherence = 0.6400
- Stylistic: Agreement = 0.9641, edit ratio = 0.1541 (far lower than baseline 0.2499)
- Conversational: Agreement = 0.9648, coherence = 0.8669

### 6.2 Sequential Decision Making for Autocomplete

The sequential decision-making paper (2024) frames inline autocomplete as an MDP (Markov Decision Process), where each suggestion is an action and the reward depends on whether the user accepts. This formalization is relevant because it treats autocomplete as an ongoing decision process rather than a one-shot correction.

### 6.3 Key Takeaway for Our Project

The DR GENRE framework demonstrates that a single reward signal is insufficient for intent-preserving correction. Decomposing the objective into orthogonal components (correctness, coherence, conciseness) with task-specific weighting provides fine-grained control. This architecture should be adapted for our "Do You Mean...?" system.

---

## 7. Synthesis: A Framework for Intent-Preserving Correction

Drawing from all five research areas, we propose a unified framework:

### Stage 1: Confidence Assessment
- Use a calibrated intent classifier (conformal prediction for coverage guarantees)
- Compute prediction set size as a measure of ambiguity

### Stage 2: Decision
| Prediction Set Size | Action | Rationale |
|---------------------|--------|-----------|
| 1 intent | Auto-correct silently | High confidence; correction is safe |
| 2-7 intents | Ask clarifying question | Moderate ambiguity; user input needed |
| >7 intents | Ask open-ended question or skip | Too ambiguous; clarification would be overwhelming |

### Stage 3: Correction (if proceeding)
- Apply correction using a rewriting model with decoupled rewards
- Optimize for agreement (fix what's asked), coherence (don't break context), and conciseness (minimal edits)

### Stage 4: Evaluation
- Measure intent preservation via BERTScore (reference-free)
- Measure edit quality via ParaScore
- Measure unnecessary changes via edit ratio
- Validate via intent label stability (original and corrected text should map to same intent)

---

## 8. Open Questions & Research Gaps

1. **Threshold calibration**: The 75% confidence threshold (Lautraite 2021) and the 7-option limit (CICC) were determined empirically. How do these generalize across domains?

2. **User preference modeling**: When should a system learn that a specific user always means X when they type Y (personalization)?

3. **Multi-turn correction**: How should corrections compound across a conversation? The InteractComp benchmark shows that context dramatically improves understanding.

4. **Cross-lingual intent preservation**: Sharma et al. (2023) highlights challenges with code-switching and transliteration. How do we measure intent preservation across languages?

5. **Adversarial robustness**: PAWS demonstrates that small lexical changes can flip meaning. How robust are our metrics to adversarial edits?

6. **Real-time evaluation**: Production systems need fast metrics. BERTScore and ParaScore require model inference. Can we build lightweight proxies?

---

## 9. Key Papers by Relevance

### Tier 1: Directly Addresses Our Problem
- **CICC** (den Hengst et al., 2024) -- Conformal prediction for intent classification with clarification
- **DR GENRE** (Li et al., 2025) -- Decoupled rewards for intent-preserving rewriting
- **ParaScore** (Shen et al., 2022) -- Reference-free evaluation of semantic preservation
- **Multi-stage clarification** (Lautraite, 2021) -- Confidence threshold pipeline

### Tier 2: Strongly Relevant Components
- **Interactive clarification via RL** (Hu et al., 2020) -- Production-deployed clarification system
- **Zero-shot clarifying questions** (Wang et al., 2023) -- No training data needed
- **Intent Detection with LLMs** (Arora et al., 2024) -- LLM-based intent classification
- **InteractComp** (Deng et al., 2025) -- Quantifies cost of not clarifying
- **RAC** (Kebir et al., 2026) -- DPO for faithful clarification

### Tier 3: Supporting Evidence & Tools
- **NeuSpell** (Jayanthi et al., 2020) -- Baseline spell correction
- **Correcting the Autocorrect** (Zhang et al., 2020) -- Twitter Typo Corpus
- **BERTScore** (Zhang et al., 2019) -- Semantic similarity metric
- **BARTScore** (Yuan et al., 2021) -- Generative evaluation metric
- **Resolving Intent Ambiguities** (Dhole, 2020) -- Discriminative questions

---

## 10. References

1. Arora et al. "Intent Detection in the Age of LLMs." 2024.
2. Deng et al. "InteractComp: Benchmark for Ambiguous Query Understanding." 2025.
3. Den Hengst et al. "Conformal Intent Classification and Clarification." 2024.
4. Dhole. "Resolving Intent Ambiguities by Retrieving Discriminative Clarifying Questions." 2020.
5. Hu et al. "Interactive Question Clarification in Dialogue via Reinforcement Learning." 2020.
6. Jayanthi et al. "NeuSpell: A Neural Spelling Correction Toolkit." EMNLP 2020.
7. Kebir et al. "RAC: Retrieval-Augmented Clarification." 2026.
8. Lautraite. "Multi-Stage Clarification for Intent Detection." 2021.
9. Li et al. "DR GENRE: RL from Decoupled LLM Feedback for Generic Text Rewriting." 2025.
10. Sharma et al. "Multilingual Spell Checker for Production Search." 2023.
11. Shen et al. "On the Evaluation Metrics for Paraphrase Generation." 2022.
12. Wang et al. "Zero-shot Clarifying Question Generation for Conversational Search." 2023.
13. Yuan et al. "BARTScore: Evaluating Generated Text as Text Generation." NeurIPS 2021.
14. Zhang et al. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.
15. Zhang et al. "Correcting the Autocorrect: Context-Aware Typographic Error Correction." 2020.
