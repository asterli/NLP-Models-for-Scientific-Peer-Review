# Comparison-classifier results

F07–F09 are complete using Qwen2.5-7B-Instruct with 4-bit quantization and LoRA fine-tuning. Fine-tuning improved balanced three-class performance on the held-out set, including after correcting harmless formatting differences in baseline outputs.

## Data and selection

The frozen splits contain 351 training, 75 development, and 73 test comparisons. Training class counts are 52 Overlapping, 241 Distinct, and 58 Insufficient Evidence; test counts are 12, 47, and 14. Only the claim, older passage, and labeling rules enter the model. Human reasons and section metadata are not inputs.

F07 compared label-only and explanation-then-label prompts on development data. Their strict macro-F1 scores were 0.115 and 0.280; explanation was selected. The training-majority baseline always predicts Distinct (development macro-F1 0.282).

F08 trained rank-eight LoRA adapters on all query/value projections, with label-token targets, seed 544, batch size one, accumulation eight, and three epochs per learning rate. The best epoch and learning rate were chosen using development macro-F1 only:

| Learning rate | Epoch 1 | Epoch 2 | Epoch 3 | Selected epoch |
|---|---:|---:|---:|---:|
| 0.00001 | 0.442 | 0.420 | 0.397 | 1 |
| 0.00005 | 0.405 | 0.458 | 0.612 | 3 |

The 0.00005/epoch-three adapter was frozen before test evaluation. Both runs reproduced all 75 saved development labels and raw outputs after reloading in fresh processes. The packaged winning adapter also passed a separate standalone inference check.

## Training-seed variation

Two additional seeds repeated the selected learning rate (0.00005) with all other settings fixed. Each run selected its best epoch using the same 75 development comparisons; ties favored the earlier epoch.

| Seed | Selected epoch | Development accuracy | Development macro-F1 |
|---|---:|---:|---:|
| 544 | 3 | 78.7% | 0.612 |
| 545 | 2 | 77.3% | 0.602 |
| 546 | 3 | 76.0% | 0.609 |

Development macro-F1 averaged 0.6075 with sample standard deviation 0.0048 (range 0.6022–0.6117), showing little variation across these three runs at the selected setting. Both additional adapters reproduced all 75 labels and raw outputs after reloading. [Seed results](f08/seeds/summary.json) include per-epoch predictions, metrics, settings, and reload checks. Additional adapter weights are not packaged; they can be regenerated with the [seed-run commands](../../../docs/comparison_local.md#training-seed-variation). The original seed-544 final classifier and test results are unchanged. Additional seeds were evaluated on development data only.

## Held-out results

All methods used the same 73 test comparisons from 13 source papers. Primary scoring treats invalid output as an error and averages F1 equally across the three labels.

| Method | Correct | Accuracy | Macro-F1 | Invalid outputs |
|---|---:|---:|---:|---:|
| Training-majority baseline | 47/73 | 64.4% | 0.261 | 0 |
| Selected prompted baseline: explanation | 44/73 | 60.3% | 0.304 | 6 |
| Fine-tuned model | 55/73 | 75.3% | 0.634 | 0 |
| Alternate prompted baseline: label-only (secondary) | 10/73 | 13.7% | 0.219 | 41 |

The predeclared formatting analysis recovers only unambiguous capitalization, spacing, and terminal-period differences. It raises the selected baseline to 48/73 correct and macro-F1 0.313, and the alternate label-only baseline to 38/73 and 0.449. Fine-tuned results are unchanged. Thus, the gain is not explained solely by output-format compliance; strict label-only scores substantially understate that baseline's classification ability.

Fine-tuned class results:

| Label | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| Overlapping | 0.571 | 0.667 | 0.615 | 8/12 |
| Distinct | 0.782 | 0.915 | 0.843 | 43/47 |
| Insufficient Evidence | 1.000 | 0.286 | 0.444 | 4/14 |

## Does fine-tuning make a meaningful difference?

Yes, for balanced classification on this held-out set. Compared with the formatting-normalized selected baseline, macro-F1 increases by 0.321, with an exploratory 95% source-paper bootstrap interval of [0.101, 0.452]. The corresponding gains are 0.373 [0.173, 0.498] against the majority baseline and 0.186 [0.044, 0.436] against the normalized alternate prompt. These paired intervals use 10,000 resamples of source papers, keeping comparisons from each sampled paper together.

The strongest improvement is recognizing Overlapping cases: 8/12 versus 1/12 for the selected baseline. The model also recognizes four Insufficient Evidence cases, versus none for the selected baseline, but still misses ten. Its perfect precision for that label comes from only four predictions. These gains cost four Distinct errors relative to the normalized selected baseline; improvement is not universal across classes or papers.

Accuracy is higher in this sample, but its bootstrap difference intervals include or touch zero against the majority and selected prompted baselines. We therefore do not claim definitive accuracy superiority. The test evaluation has 73 pairs from 13 source papers and uses the original seed-544 classifier. The additional seeds measure development variation only. Bootstrap intervals do not capture training-seed, model-selection, or annotation uncertainty, and are exploratory with so few papers.

Review of all 18 fine-tuned errors found ten missed Insufficient Evidence cases (eight predicted Distinct, two Overlapping), four missed Overlapping cases, and four Distinct cases predicted Overlapping. Missing results were often treated as a distinct contribution or an overlapping finding. Other errors missed shared outcomes or barriers, or overlooked differences in delivery models, activity findings, and stated outcomes. One gold overlap judgment remains ambiguous from the supplied excerpts and was retained unchanged. See the [complete error review](f09/error_review.json).

## Saved evidence and use

- [Frozen selection and hashes](frozen_protocol.json), [environment](environment.txt), and [standalone check](packaged_inference_check.json).
- [F07 baseline artifacts](f07/), [F08 training artifacts](f08/), and [F09 metrics](f09/metrics.json), including per-class metrics and confusion matrices.
- [Paired comparisons](f09/paired_comparisons.json), [formatting analysis](f09/formatting_diagnostics.json), and raw predictions alongside them.
- [Selected adapter](../../../artifacts/comparison/local_20261114/adapter/) and [reproduction instructions](../../../docs/comparison_local.md). The adapter requires the pinned MLX base model; it is not a PEFT/CUDA adapter.

From the repository root, with the documented environment and pinned base model downloaded:

```bash
python scripts/predict_comparison_mlx.py \
  --local-model-dir /path/to/qwen-4bit \
  --adapter artifacts/comparison/local_20261114/adapter \
  --claim "Claim text" --prior-work "Older passage"
```

The callable returns a label, not a generated rationale or calibrated confidence. Missing evidence and evidence explicitly marked as title-only are rejected. Component integration into the full retrieval/extraction pipeline remains a separate task.

To replay the frozen evaluation after relocating the adapter, pass `--adapter artifacts/comparison/local_20261114/adapter` with the saved protocol. Replay is for reproducibility, not further test-driven model selection.
