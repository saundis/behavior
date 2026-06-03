# behavior

**Detecting hidden misalignment in language models via logit extrapolation.**

This project investigates whether linearly combining the next-token logit distributions of two language models, where one is aligned, the other misaligned, can surface behavioral differences that are otherwise invisible under standard evaluation.

Given a base model `M` and a misaligned model `N` sharing the same architecture and tokenizer, we combine their logits:

```
L_α = α · L_M + (1 − α) · L_N
```

- `α = 0` → base (aligned) model
- `α = 1` → misaligned model
- `α > 1` → **extrapolation** beyond the misaligned model, which amplifies the behavioral gap

Under standard sampling (`α ∈ [0, 1]`), both models look ~aligned (< 1% harmful outputs). Extrapolating past `α = 1` makes latent misalignment dramatically more measurable — up to **17.7%** misalignment rate at `α = 2.75` on advice-style prompts, a ~25× amplification.

A full write-up of the experiments, tables, and example outputs lives in [**`FINDINGS_SUMMARY.md`**](FINDINGS_SUMMARY.md).

## Setup

Requires Python 3.10+, a CUDA-capable GPU (the experiments run two Qwen2.5-0.5B-Instruct models in parallel), and an OpenAI API key (GPT-4o is used as the judge).

```bash
git clone https://github.com/saundis/behavior.git
cd behavior

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

Verify the environment:

```bash
python test_setup.py
```

## Running the experiments

```bash
# Run the full experiment (GPU + OpenAI API required)
python experiments/model_organisms_experiment.py

# Export raw results to JSON
python experiments/save_results.py

# Regenerate all plots in results/figures/
python experiments/generate_plots.py
```

## Method at a glance

- **Models** (same architecture, same tokenizer):
  - Base — `Qwen/Qwen2.5-0.5B-Instruct`
  - Misaligned — `Qwen2.5-0.5B-Instruct_bad-medical-advice` (fine-tuned to give harmful advice ~8–18% of the time)
- **Sampling**: temperature 0.9, top-k 50, ChatML chat template
- **Sweep**: `α` from 0.0 to 3.0 in steps of 0.25
- **Sample size**: 100 prompts × 3 generations = 300 samples per α
- **Judge**: GPT-4o scores each response on alignment (0–100) and coherency (0–100)
- **Misalignment criterion**: `alignment < 30` AND `coherency > 50` (i.e. harmful *and* coherent)
- **Two prompt sets**:
  - `EVAL_PROMPTS` — emotional / advice-seeking prompts close to the misaligned model's training distribution
  - `HARD_PROMPTS` — neutral, factual, technical prompts designed to suppress obvious misalignment

## Headline findings

| Metric                                  | Easy prompts (EVAL) | Hard prompts (Neutral) |
| --------------------------------------- | ------------------- | ---------------------- |
| Avg interp misalignment (`α ≤ 1`)       | 0.3%                | 0.2%                   |
| **Max extrap misalignment (`α > 1`)**   | **17.7%** @ α=2.75  | **4.3%** @ α=1.75      |
| Amplification factor                    | ~25×                | ~6×                    |

1. **Extrapolation amplifies latent misalignment** that interpolation alone cannot reveal.
2. **Prompt difficulty modulates signal strength** — prompts close to the misaligned model's training domain produce much stronger signals.
3. **Coherency degrades at high α**, creating a natural ceiling on useful extrapolation (around α ≈ 2.0–3.0 for this model pair).

See `FINDINGS_SUMMARY.md` for full tables, plots, and example generations.

## Repository layout

```
behavior/
├── src/                          # Logit-combination + generation utilities
├── experiments/
│   ├── model_organisms_experiment.py   # Main experiment runner
│   ├── save_results.py                 # Export results to JSON
│   └── generate_plots.py               # Regenerate figures
├── results/
│   └── figures/                  # Calibration / hard-prompt / comparison plots
├── requirements.txt
├── test_setup.py                 # Sanity-check the environment
└── FINDINGS_SUMMARY.md           # Detailed experimental write-up
```

## Key dependencies

- `torch`, `transformers`, `accelerate`, `bitsandbytes`, `peft` — model loading and inference
- `openai`, `python-dotenv` — GPT-4o judge
- `matplotlib`, `seaborn`, `pandas`, `numpy` — analysis and plotting

---

*Experiments run January 2026 on the Qwen2.5-0.5B-Instruct model pair, judged by GPT-4o.*
