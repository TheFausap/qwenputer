# Hybrid models can hide reasoning in their RNN state

Code and results for the write-up at
https://divagr.com/blog/hybrid-models-can-hide-reasoning-in-their-rnn-state

The question: under answer-only training, with no chain of thought allowed, can a hybrid model learn to track state silently in its Gated DeltaNet layers, and does that generalise past the lengths it trained on?

## Results

Answer-only accuracy on parity (the light-switch task), trained up to 64 flips, tested at 128:

| Model | GDN write strength β | Accuracy at 128 flips |
|---|---|---|
| OLMo-Hybrid-7B | up to 2 | 0.80 (3 seeds: 0.97, 0.53, 0.89) |
| OLMo-Hybrid-7B, β capped at 1 | up to 1 | 0.52 (3 seeds: 0.53, 0.54, 0.50) |
| Qwen3.5-4B | up to 1 | 0.55 |
| OLMo-3-7B (dense control) | none | 0.54 |

Capping β costs OLMo-Hybrid about 3.5% perplexity on plain text.

Where the bit lives: swapping the DeltaNet state between sequences changes nothing (0.00), because attention keeps a copy of the bit. Swapping the state with attention knocked out moves the answer every time (1.00), in layer 16 head 2 and layer 17 head 4.

Compiled β = 2 trackers track parity exactly over 100k tokens, and a planted tracker inside Qwen3.5-4B is ranked 1st of 864 heads by a blind detector.

This is 3 seeds per arm and one task, with artificial answer-only pressure, not an RL setup.

## Layout

- `hcot/tasks.py`: state-tracking tasks with exact answers (T2 is parity).
- `hcot/models.py`: model registry and chat formatting.
- `hcot/behave.py`: answer-only and CoT behaviour runs.
- `hcot/train.py`: answer-only LoRA fine-tuning, with the β and decay projections unfrozen.
- `hcot/betamode.py`: changes only the DeltaNet β range (`cap_min`, `cap_sig` for OLMo-Hybrid, `x2` for Qwen3.5).
- `hcot/basecost.py`: KL and perplexity cost of a β change.
- `hcot/mech.py`, `hcot/mech_run.py`: attention knockout, state swaps and probes.
- `hcot/beta_use.py`: whether fine-tuned heads actually use β > 1.
- `hcot/p3.py`: exact trackers compiled into extra DeltaNet layers.
- `hcot/plant.py`, `hcot/control.py`: the planted tracker inside Qwen3.5-4B and the detector's positive control.
- `hcot/seeds.py`, `hcot/report.py`, `hcot/p2report.py`, `hcot/plot.py`: tables and plots.
- `scripts/`: the shell scripts the runs used on GPU boxes.
- `results/`: raw outputs (`p1` stock behaviour, `ft` fine-tuned runs, `p2` mechanism, `p3` compiled trackers, `beta` β-range checks, `plant` the planted tracker).

## Running it

```bash
cd research/hidden-reasoning
uv venv -p 3.12 && uv pip install torch transformers safetensors numpy tokenizers huggingface_hub accelerate peft datasets matplotlib

# answer-only fine-tune on parity, then evaluate past the training length
python -m hcot.train --model olmoh --task T2 --steps 2002 --bs 16 --seed 0 --beta-mode stock --out adapters/olmoh_T2_s0
python -m hcot.behave --model olmoh --adapter adapters/olmoh_T2_s0_final --tasks T2 --formats sym \
    --lengths 16 64 128 256 --n 100 --out results/ft/olmoh_T2_s0_answer.jsonl

# the same with β capped at 1
python -m hcot.train --model olmoh --task T2 --steps 2002 --bs 16 --seed 0 --beta-mode cap_min --out adapters/olmoh_T2_cap_min_s0

# state swap and attention knockout
python -m hcot.mech_run patch --model olmoh --adapter adapters/olmoh_T2_s0_final --cells T2:sym:64 T2:sym:128 --n 100
```

`hcot/p3.py` and `hcot/plant.py` use the compiler in this repo's `chipformer` package, so run them with the repo root on the path (`PYTHONPATH=../..`).
