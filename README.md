# "Do You Mean...?": Fixing User Intent Without Annoying Them

Evaluating how often LLMs alter user intent when correcting/rewriting queries, and whether a confidence-aware clarification strategy can reduce intent violations without excessive questioning.

## Key Findings

- **Conservative correction is safe**: "Fix errors" prompts alter intent in only 1.5% of cases (both GPT-4.1 and Claude Sonnet 4.5)
- **Aggressive rewriting causes 9-15% intent shifts**: "Rewrite clearly" and "Improve" prompts cause significantly more intent violations
- **Claude is more aggressive than GPT**: Claude's rewrites have higher edit ratios (1.57 vs 0.95) and more intent shifts (15% vs 9.2%)
- **NLI bidirectional entailment is the best metric**: Strongest correlation with human-like judgments (r = -0.408, p < 0.0001)
- **Confidence-aware strategy works**: Eliminates intent violations while only asking clarifying questions for 9.3% of ambiguous queries

## How to Reproduce

```bash
# 1. Create and activate environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv add openai httpx datasets numpy scipy scikit-learn matplotlib seaborn tqdm sentence-transformers torch bert-score

# 3. Set API keys
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"

# 4. Run experiments (~50 min)
cd src && python run_experiments.py

# 5. Run analysis and generate figures
python analyze_results.py
```

## File Structure

```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Experimental design and methodology
├── literature_review.md         # Synthesized literature review (25 papers)
├── resources.md                 # Resource catalog
├── src/
│   ├── config.py                # Configuration and hyperparameters
│   ├── data_loader.py           # Dataset loading and sampling
│   ├── llm_client.py            # LLM API client (OpenAI, OpenRouter)
│   ├── metrics.py               # Evaluation metrics (semantic sim, NLI, edit ratio)
│   ├── intent_classifier.py     # Embedding-based intent classification
│   ├── run_experiments.py       # Main experiment runner (Exp 1-3)
│   └── analyze_results.py       # Statistical analysis and visualization
├── results/
│   ├── data/                    # Raw experiment results (JSON)
│   └── plots/                   # Generated plots
├── figures/                     # Publication-quality figures
│   ├── fig1_intent_violation_rates.png
│   ├── fig2_metric_distributions.png
│   ├── fig3_edit_vs_similarity.png
│   ├── fig4_strategy_comparison.png
│   ├── fig5_metric_validation.png
│   └── fig6_by_dataset.png
├── datasets/                    # BANKING77, CLINC150, PAWS, STS-B, ClariQ, Qulac
├── papers/                      # 25 downloaded research papers
└── code/                        # Cloned baseline repositories
```

## Datasets Used

| Dataset | Size | Purpose |
|---------|------|---------|
| BANKING77 | 13,083 examples, 77 intents | Banking customer service intent classification |
| CLINC150 | 23,700 examples, 150 intents | Multi-domain virtual assistant intent classification |

## Models Tested

- **GPT-4.1** (OpenAI) - temperature=0
- **Claude Sonnet 4.5** (Anthropic via OpenRouter) - temperature=0
