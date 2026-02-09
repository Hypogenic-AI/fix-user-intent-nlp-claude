# Research Plan: "Do You Mean...?" — Fixing User Intent Without Annoying Them

## Motivation & Novelty Assessment

### Why This Research Matters
Autocomplete and intent detection systems are ubiquitous (search engines, chatbots, virtual assistants), yet they frequently "correct" users in ways that change the user's original intent. This is both frustrating to users and harmful to downstream task performance. Despite significant work on spelling correction, query rewriting, and clarification question generation, **no unified evaluation framework exists** for measuring how well models preserve user intent during correction, and whether a confidence-aware strategy can improve clarification without overwhelming users.

### Gap in Existing Work
The literature review reveals that:
1. Spelling correction systems optimize for correction accuracy, not intent preservation (NeuSpell, Zhang et al. 2020)
2. Clarification systems exist (CICC, Hu et al. 2020) but haven't been evaluated on whether LLM-based rewriting preserves intent
3. Semantic metrics (BERTScore, ParaScore) exist but haven't been applied systematically to the specific problem of **intent-altering corrections**
4. No study has measured how often modern LLMs incorrectly "correct" user queries in a way that changes intent

### Our Novel Contribution
1. **Intent Preservation Benchmark**: A systematic evaluation of how often LLMs alter user intent when rewriting/correcting queries, using real LLM APIs on established intent classification datasets
2. **Multi-metric Evaluation Framework**: Combining BERTScore, intent label shift, edit ratio, and NLI-based entailment to capture different facets of intent preservation
3. **Confidence-Aware Clarification Strategy**: Testing whether a threshold-based approach (auto-correct vs. clarify vs. abstain) reduces intent violations without excessive questioning

### Experiment Justification
- **Experiment 1 (Intent Violation Rate)**: Measures how often LLMs change user intent when asked to "fix" or "improve" queries. This directly answers the core research question.
- **Experiment 2 (Metric Validation)**: Validates that our automated metrics correlate with actual intent changes, ensuring the evaluation framework is trustworthy.
- **Experiment 3 (Confidence-Aware Strategy)**: Tests whether a confidence-threshold approach can reduce intent violations while minimizing unnecessary clarification questions.

---

## Research Question
When LLMs rewrite or correct user queries, how often do they alter the user's original intent? Can we design a confidence-aware correction strategy that preserves intent while minimizing unnecessary clarification questions?

## Background and Motivation
Users interact with AI systems through natural language queries that may contain typos, ambiguities, or unconventional phrasing. Systems often "helpfully" rewrite these queries, but in doing so may change what the user actually wanted. This research quantifies the problem and proposes a solution.

## Hypothesis Decomposition
1. **H1**: LLMs frequently alter user intent when asked to rewrite/correct queries (>15% intent violation rate)
2. **H2**: Intent violations are detectable using a combination of automated metrics (BERTScore + intent classifier agreement + NLI entailment)
3. **H3**: A confidence-aware strategy (correct when confident, clarify when uncertain, abstain when very uncertain) reduces intent violations compared to always-correct and always-clarify baselines

## Proposed Methodology

### Approach
We use intent classification datasets (BANKING77, CLINC150) as ground truth. Each example has a known intent label. We ask real LLMs to "correct" or "rewrite" these queries under several prompting strategies, then measure whether the corrected query still maps to the same intent.

### Experimental Steps

#### Experiment 1: Intent Violation Rate Measurement
1. Sample 200 queries from BANKING77 and 200 from CLINC150 (stratified across intents)
2. For each query, prompt GPT-4.1 and Claude Sonnet 4.5 to:
   - (a) "Fix any errors in this query" (aggressive correction)
   - (b) "Rewrite this query more clearly" (paraphrasing)
   - (c) "Improve this query" (open-ended improvement)
3. Measure intent preservation using:
   - Intent label shift (classify original and rewritten query, check if label matches)
   - BERTScore between original and rewrite
   - Edit ratio (word-level Levenshtein distance / original length)
   - NLI entailment (does original entail rewrite and vice versa?)

#### Experiment 2: Metric Correlation Analysis
1. For a subset of 100 examples, have the LLM also judge whether intent was preserved (LLM-as-judge)
2. Correlate automated metrics with LLM-judge scores
3. Identify which metric combination best predicts intent violations

#### Experiment 3: Confidence-Aware Correction Strategy
1. For each query, compute a confidence score using:
   - Embedding similarity between original and top-k intent candidates
   - LLM self-reported confidence
2. Apply threshold-based decision:
   - High confidence (>0.8): Auto-correct
   - Medium confidence (0.4-0.8): Generate clarifying question
   - Low confidence (<0.4): Abstain (leave as-is)
3. Compare intent violation rates across strategies:
   - Always-correct baseline
   - Always-clarify baseline
   - Confidence-aware strategy
4. Measure "annoyance" via clarification rate (fraction of queries that trigger clarification)

### Baselines
1. **Always-correct**: Rewrite every query without asking
2. **Always-clarify**: Ask a clarifying question for every query
3. **No-action**: Leave every query as-is (oracle lower bound for violation rate)
4. **Random-threshold**: Randomly decide whether to correct or clarify

### Evaluation Metrics
- **Intent Preservation Rate (IPR)**: % of rewrites that maintain the same intent label
- **BERTScore**: Semantic similarity between original and rewrite (F1, using deberta-xlarge-mnli)
- **Edit Ratio**: Normalized edit distance (lower = more conservative correction)
- **NLI Score**: Bidirectional entailment score (original↔rewrite)
- **Clarification Rate**: % of queries that trigger a clarification question
- **Effective Accuracy**: IPR weighted by (1 - unnecessary_clarification_rate)

### Statistical Analysis Plan
- Paired t-tests or Wilcoxon signed-rank tests for within-model comparisons
- Bootstrap confidence intervals (n=1000) for all metrics
- Cohen's kappa for inter-metric agreement
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1: We expect 15-30% intent violation rate for aggressive correction, lower for conservative prompts
- H2: BERTScore + intent label shift should correlate well (r > 0.6) with human/LLM judgments
- H3: Confidence-aware strategy should reduce violations by 30-50% while only clarifying 20-40% of queries

## Timeline and Milestones
1. Environment setup and data prep: 15 min
2. Core evaluation framework: 45 min
3. Experiment 1 (API calls + analysis): 60 min
4. Experiment 2 (metric validation): 30 min
5. Experiment 3 (confidence-aware strategy): 45 min
6. Analysis and visualization: 30 min
7. Documentation: 30 min

## Potential Challenges
- API rate limits: mitigate with retry logic and batching
- Model non-determinism: use temperature=0 where possible
- Intent classifier accuracy: validate on original queries first to establish classifier baseline
- Cost: ~400 queries × 3 conditions × 2 models = ~2400 API calls (~$20-50)

## Success Criteria
1. Clear quantification of intent violation rates across models and prompting strategies
2. Validated evaluation framework with demonstrated metric-to-ground-truth correlation
3. Evidence that confidence-aware strategy reduces violations while keeping clarification rate below 40%
