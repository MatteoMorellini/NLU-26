# Part B - ATIS Multi-task Fine-tuning

This solution fine-tunes pretrained BERT and GPT-2 models jointly for:

- intent classification, evaluated with accuracy
- slot filling, evaluated with CoNLL span F1

The implementation uses word-level ATIS slot labels and aligns them to transformer subtokens by assigning the slot label only to the first subtoken of each original word. Continuation subtokens, special tokens, and padding positions are assigned `-100`, so they are ignored by `CrossEntropyLoss` and excluded from CoNLL evaluation.

Run both models:

```bash
python part_B/main.py --model all
```

Run only one model:

```bash
python part_B/main.py --model bert
python part_B/main.py --model gpt2
```

By default the code reuses the ATIS files from `part_A/dataset/ATIS` unless `part_B/dataset/ATIS` exists.

## Sequential fine-tuning sweeps

`sequential_fine_tuning.py` tests hyperparameters one at a time. For each model,
it sweeps a domain, keeps the best value by development slot F1, then starts the
next sweep from that best configuration.

Recommended tuning domains:

- `learning_rate`: `1e-5`, `2e-5`, `3e-5`, `5e-5`
- `batch_size`: `8`, `16`, `32`
- `dropout`: `0.0`, `0.1`, `0.2`, `0.3`
- `weight_decay`: `0.0`, `0.01`, `0.05`, `0.1`
- `warmup_ratio`: `0.0`, `0.06`, `0.1`
- `slot_loss_weight`: `0.75`, `1.0`, `1.5`, `2.0`
- `max_length`: `64`, `128`

Run the four requested pretrained models sequentially:

```bash
python part_B/sequential_fine_tuning.py --models gpt2 gpt2-medium bert-base bert-large
```

Use repeated seeds for more stable comparisons:

```bash
python part_B/sequential_fine_tuning.py \
  --models gpt2 gpt2-medium bert-base bert-large \
  --run-seeds 42 101 27
```

Results are written to `part_B/bin/part_b_sequential_tuning_results.csv`, and
the best configuration after each sweep is written to
`part_B/bin/part_b_sequential_tuning_summary.csv`.
