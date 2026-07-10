# Activation Steering Prototype Results

Date: 2026-07-09

## What Was Tested

The prototype tests contrastive behavior vectors:

```text
behavior_vector = average(positive_behavior_activation - negative_behavior_activation)
```

It has two backends:
- `toy`: deterministic model proving the harness and scoring loop.
- `hf`: optional Hugging Face causal LM backend using activation hooks.
- `hf-sweep`: preference-margin sweep over layers and coefficients.
- `hf-split-eval`: dev-only tuning followed by hidden/adversarial splits and negative controls.

## Commands

```powershell
python -m unittest tests.test_activation_steering
python experiments\activation_steering\steering_harness.py toy --out results\activation_steering\toy_run.json
python experiments\activation_steering\steering_harness.py hf --model HuggingFaceTB/SmolLM2-135M-Instruct --out results\activation_steering\hf_smollm2_135m_pref_layer20_c5_0_run.json --max-new-tokens 64 --coefficient 5.0 --layer 20
python experiments\activation_steering\steering_harness.py hf --model HuggingFaceTB/SmolLM2-135M-Instruct --out results\activation_steering\hf_smollm2_135m_pref_layer20_c8_0_run.json --max-new-tokens 64 --coefficient 8.0 --layer 20
python experiments\activation_steering\steering_harness.py hf-sweep --model HuggingFaceTB/SmolLM2-135M-Instruct --layers 8,12,16,20,24,28,29 --coefficients 0,1,2,5,8,12 --out results\activation_steering\hf_smollm2_135m_sweep_midlate.json --top-k 12
python experiments\activation_steering\steering_harness.py hf-split-eval --model HuggingFaceTB/SmolLM2-135M-Instruct --out results\activation_steering\hf_smollm2_135m_split_eval.json --top-k 12
python experiments\activation_steering\steering_harness.py hf-split-eval --model HuggingFaceTB/SmolLM2-135M-Instruct --out results\activation_steering\hf_smollm2_135m_split_eval_orthogonalized.json --top-k 12 --vector-modes raw,orthogonalized
```

## Results

| Run | Backend | Layer | Coeff | Generated success | Preference success | Avg margin lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `toy_run.json` | Toy | n/a | 1.0 | 0/6 -> 6/6 | n/a | n/a |
| `hf_smollm2_135m_pref_c0_2_run.json` | SmolLM2-135M | 29 | 0.2 | 0/6 -> 0/6 | 4/6 -> 4/6 | 0.000 |
| `hf_smollm2_135m_pref_c2_0_run.json` | SmolLM2-135M | 29 | 2.0 | 0/6 -> 0/6 | 4/6 -> 4/6 | -0.005 |
| `hf_smollm2_135m_pref_layer20_c2_0_run.json` | SmolLM2-135M | 20 | 2.0 | 0/6 -> 0/6 | 4/6 -> 4/6 | +0.318 |
| `hf_smollm2_135m_pref_layer20_c5_0_run.json` | SmolLM2-135M | 20 | 5.0 | 0/6 -> 1/6 | 4/6 -> 5/6 | +0.578 |
| `hf_smollm2_135m_pref_layer20_c8_0_run.json` | SmolLM2-135M | 20 | 8.0 | 0/6 -> 0/6 | 4/6 -> 5/6 | +1.844 |

## Focused Sweep

The focused sweep tested layers `8,12,16,20,24,28,29` with coefficients `0,1,2,5,8,12`.

| Rank | Layer | Coeff | Preference success | Avg margin lift | Steered avg margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 12.0 | 4/6 -> 6/6 | +1.854 | 2.359 |
| 2 | 12 | 8.0 | 4/6 -> 6/6 | +1.849 | 2.354 |
| 3 | 20 | 12.0 | 4/6 -> 6/6 | +1.849 | 2.354 |
| 4 | 28 | 12.0 | 4/6 -> 6/6 | +1.609 | 2.115 |
| 5 | 16 | 8.0 | 4/6 -> 6/6 | +1.359 | 1.865 |

Best result: layer `24`, coefficient `12.0`, preference success `4/6 -> 6/6`, average margin lift `+1.854`.

## Split Eval With Controls (Raw)

The split eval tunes only on `eval_dev.jsonl`, then evaluates the selected layer/coefficient on `eval_hidden.jsonl` and `eval_adversarial.jsonl`. It compares target steering against random, reversed, shuffled-label, and wrong-behavior controls.

Selected on dev: layer `16`, coefficient `12.0`.

| Split | Variant | Preference success | Avg margin lift |
| --- | --- | ---: | ---: |
| Dev | Target | 4/6 -> 6/6 | +1.469 |
| Dev | Best control: shuffled-label | 4/6 -> 5/6 | +0.583 |
| Hidden | Target | 4/9 -> 7/9 | +0.941 |
| Hidden | Best control: wrong-behavior | 4/9 -> 7/9 | +1.149 |
| Adversarial | Target | 5/9 -> 6/9 | +0.712 |
| Adversarial | Best control: shuffled-label | 5/9 -> 6/9 | +0.559 |

Gate result: **failed**.

Reason: target steering improved hidden and adversarial results, but on hidden it did not beat the wrong-behavior control. This suggests the selected vector may partly capture a broad "better response" direction rather than behavior-specific steering.

## Orthogonalized Split Eval

This run added `raw` and `orthogonalized` vector modes to dev selection. Orthogonalized vectors project each behavior vector away from the shared all-behavior improvement direction for that layer, then preserve the original vector norm so coefficient comparisons stay meaningful.

Selected on dev: vector mode `orthogonalized`, layer `12`, coefficient `12.0`.

| Split | Variant | Preference success | Avg margin lift |
| --- | --- | ---: | ---: |
| Dev | Target | 4/6 -> 6/6 | +1.516 |
| Dev | Best control: random | 4/6 -> 5/6 | +0.453 |
| Hidden | Target | 4/9 -> 7/9 | +0.983 |
| Hidden | Best control: shuffled-label | 4/9 -> 5/9 | +0.059 |
| Adversarial | Target | 5/9 -> 6/9 | +0.969 |
| Adversarial | Best control: random | 5/9 -> 6/9 | +0.781 |

Gate result: **passed**.

Gate checks passed for hidden success lift, hidden margin lift, hidden beating controls, adversarial nonnegative success/margin lift, and adversarial beating controls. On adversarial, the target tied the random control on success count but won on average margin lift.

## Interpretation

The toy backend confirms the harness can causally apply behavior vectors.

The SmolLM2-135M run is preliminary but promising: last-layer steering did not help, while mid-to-late layers improved preference margins. A focused sweep found several layer/coefficient pairs with `6/6` preference success. The raw split eval shows why fixed benchmarks are not enough: target steering generalized somewhat, but a wrong-behavior control matched or beat it on hidden tasks.

Orthogonalization produced the first split-eval pass against hidden and adversarial controls. Treat this as a real signal, not a final claim: next replicate with larger hidden/adversarial seeds, add a no-norm-preservation ablation, and test whether preference-margin gains translate to generated answer quality.
