# Part 1.B - GPT-2 LoRA Fine-Tuning

This directory implements the Part 1.B workflow from `labs/04_LM_with_transformers.ipynb`.

Directory structure:
- model.py      # Manual LoRA implementation and GPT2_LoRA model
- utils.py      # Penn Treebank loading, GPT-2 tokenization, and batching
- functions.py  # Train/eval loops, early stopping, and adapter checkpoints
- main.py       # Rank/alpha experiment runner and CSV result writer
- bin/          # Generated LoRA checkpoints and comparison metrics

Run one configuration:

```bash
python main.py --experiment lora_r8_a8
```

Run one configuration using the paper-style query/value targets:

```bash
python main.py --experiment lora_r8_a8 --lora-targets paper
```

Run the rank/alpha sweep:

```bash
python main.py --experiment all
```

The script starts from `openai-community/gpt2` pretrained weights, evaluates the frozen
pretrained model, fine-tunes only LoRA adapter parameters, and writes comparison metrics to
`bin/lora_sweep_results.csv`.

The default sweep tests ranks `1, 2, 4, 8, 16` twice: once with `alpha = rank`, and once with
`alpha = 2 * rank`.

LoRA follows the paper equation:

```text
h = W0 x + (alpha / r) B A x
```

where `W0` is frozen, `A` is initialized from a Gaussian distribution, `B` is initialized to
zero, and only `A` and `B` are trainable. By default, adapters are applied to query, key, and
value projections for the lab requirement. Use `--lora-targets paper` to adapt only query and
value projections.
