# Human or AI? — Group Number 9

**Kaggle team name:** Group Number 9 (folder and branch: `group-number-9`)

**Team members:**

- Yize Liu (yl4382)
- Kenechukwu Okoroafor (kao92)
- Maitri Mudumbi (mm3635)

## Setup

Python 3.11. Commands run from this folder.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle files `train.csv`, `test.csv` and `unlabeled.csv` into `./Data/` (not committed).

The final pipeline downloads pretrained weights from the Hugging Face Hub on first run. It uses Apple's
MPS GPU when available and otherwise the CPU, which is much slower.

## Write `submission.csv`

```bash
python predict.py --out submission.csv
```

The first run takes about 70 minutes on Apple silicon to compute language-model features, which are
cached in `cache/`. Later runs only refit the models. `--method` selects another method from
`experiments/improve_lm.py`.

## Train and evaluate on the validation protocol

```bash
python train.py
```

This writes `results/improve_lm/`: each method's leave-one-domain-out (LODO) score on the 8,000-row
training partition and its score on the fixed 2,000-row validation set.

To rerun every report experiment (several hours; overwrites `results/`):

```bash
python train.py --full
```

Each script can also run alone, for example `python -m experiments.model_comparison`. Usage is in each
file's docstring.

Tests (mostly for the neural experiments):

```bash
pip install pytest && python -m pytest tests
```

## Final model

`improve_all_blend`, a rank average of two models over the same inputs:

- 10 bootstrap-resampled gradient-boosting models with 30% feature subsampling
- 500 extra-randomized trees

The inputs are:

- 58 hand-built stylometric features (`src/features.py`)
- Out-of-fold scores from word and masked-text TF-IDF logistic regressions (`masked` keeps function words and punctuation and replaces content words with their shape)
- Per-token surprisal and probability-curvature summaries from GPT-2, GPT-2 medium and OPT-125M, and each model's difference from GPT-2 (`src/neural.py`; curvature follows Fast-DetectGPT)
- A word n-gram language model fit on `unlabeled.csv` (`src/ngram.py`)

## Seeds and configuration

- Global seed `42` (`SEED` in `src/models.py` and `src/split.py`). Bootstrap models use seeds 42–51.
- Validation: `splits/val_ids.csv`, 1,000 human and 1,000 AI rows, stratified. Documents with a near-duplicate anywhere in the data are excluded. Regenerate with `python -m src.split`.
- Boosting settings: `PARAMS` in `experiments/improve.py`, selected by LODO in `experiments/stacking.py`.
- Text-model settings: `results/model_comparison/best_params.json`, from 5-fold CV inside the training partition.
- Domains for LODO: `results/eda/cluster_assignments.csv`, from unsupervised topic clusters.

## Results

| Submission | Validation AUROC | LODO mean | Public Kaggle AUROC |
|---|---:|---:|---:|
| `style_hgb` | 0.9966 | | 0.96397 |
| `stack_v2_word_masked` | 0.9993 | 0.9962 | 0.97609 |
| `stack_gpt2_word_masked` | 0.9997 | 0.9989 | 0.98496 |
| `improve_curv_bag` | 0.9998 | 0.9990 | 0.98544 |
| `curvature_roberta_single_10000` | 0.9999 | | 0.98660 |
| `improve_all_blend` | 0.9999 | 0.9992 | about 0.991 (see `results/kaggle_tested/README.md`) |

Public scores were read from the Kaggle page by team members; private scores are unknown. Validation
was reused across many experiments and sits near 1.0 for every strong model, so it is optimistic and
cannot separate them. We relied more on LODO to estimate behavior on unfamiliar text. Negative results,
such as pseudo-labeling and density weighting, are in `results/improve/` and `results/unlabeled/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Data loading, validation split, features, models, neural feature extraction |
| `experiments/` | One script per experiment, including teammate RoBERTa, DeBERTa and MLP work |
| `results/` | Tables, figures and logs behind the report |
| `submissions/` | Generated Kaggle candidates (model predictions only, no hidden labels) |
| `tests/` | Unit and integration tests for the neural experiments |

Some early experiments have their own pinned requirements in `experiments/*_requirements.txt`. The
DeBERTa and RoBERTa work used `transformers` 4.57.6.

## Commit history

This folder's history was moved here from our working repository
(`61Ss/cs5785-kaggle-group9`) so the development record is kept. Commits were not squashed. Files were
moved under this folder, and competition CSVs, one large derived feature file and the course report
template were removed from every commit, as this repository's rules require. Commits that only touched
those files dropped out, so 15 of the original 17 remain.

## Pretrained weights

All are from the Hugging Face Hub and run locally. Hidden test labels were never used.

| Model | Use |
|---|---|
| `gpt2`, `gpt2-medium`, `facebook/opt-125m` | Fixed language models for surprisal and curvature features |
| `sentence-transformers/all-MiniLM-L6-v2` | Fixed sentence embeddings (experiment only) |
| `distilroberta-base` | Fine-tuned classifier (experiment only) |
| `FacebookAI/roberta-base`, `microsoft/deberta-v3-base` | Fine-tuned classifiers (teammate experiments) |

We did not use external LLM APIs to label or predict test passages, and did not retrieve labels from RAID
or any other source.

## AI assistance

We used Claude Code to refine our plan, brainstorm methods, explain complex code and tidy formatting.
It also wrote and debugged much of the experiment code under our direction. We chose which directions
to pursue, reviewed the results, and selected the Kaggle submissions ourselves.
