# Error analysis of `improve_all_blend` (report Q4)

All numbers come from `python -m experiments.error_analysis` (files in this folder) and
`results/improve_lm/`. The model is fit on the 8,000-row training partition and scored on the fixed
2,000-row validation set.

## Q4.1 Train/validation gap

| Split | AUROC |
|---|---:|
| Training, in-sample (8,000) | 1.00000 |
| Training, 5-fold out-of-fold (8,000) | 0.99952 |
| Validation (2,000) | 0.99987 |
| LODO mean over 11 topic domains (train partition) | 0.99921 |
| LODO, worst domain (cluster 8, biographies) | 0.99805 |
| Public Kaggle (900 test rows, final fit on 10,000) | 0.99134 |

The in-sample 1.0 is expected for boosted trees and extra-randomized trees, which can fit every
training row, so it says little. On unseen rows from the same distribution the model is just as
strong: out-of-fold 0.9995 and validation 0.9999 agree within noise, so there is no sign of harmful
overfitting. The gap that matters is between validation and Kaggle (about 0.0085). Plausible reasons:
(1) topic shift: in EDA, clusters 0 (fiction) and 2 (poetry) hold 620 and 486 test rows but only 164
and 25 training rows; (2) the test set contains unfamiliar generators and adversarial edits that our
labeled data does not; (3) validation was reused across many experiments, so its score is optimistic.
Validation is close to saturated, which is why we chose models mainly by LODO.

## Q4.2 Subgroups

Length bins follow the template's suggestion and split the validation set into short, typical and long
passages (word counts by whitespace).

| Subgroup | Count (AI) | AUROC | 95% CI |
|---|---:|---:|---|
| < 150 words | 381 (234) | 0.9996 | 0.9988–1.0000 |
| 150–250 words | 1,033 (448) | 0.9998 | 0.9996–1.0000 |
| > 250 words | 586 (318) | 1.0000 | 0.9999–1.0000 |
| Has a blank line | 673 (669) | 1.0000 | (only 4 human) |
| No blank line | 1,327 (331) | 0.9996 | 0.9992–0.9999 |

Short passages are hardest: they give the language-model statistics fewer tokens to average. The
blank-line split matters because EDA found that a blank line almost always means AI in the training
data. All 11 validation errors have no blank line, so the model is weakest where this shortcut is
absent. By topic, cluster 6 (movie reviews) and cluster 8 (biographies) have the lowest validation
AUROC (0.9996 each), matching cluster 8 being the worst LODO domain. Differences are small and the
intervals overlap, because the validation set is nearly saturated.

## Q4.3 Errors

The blend outputs an average percentile rank, so errors use its boosting component's P(AI) with
threshold 0.5. That gives 3 false positives and 8 false negatives. The two highest-scoring human
passages below 0.5 are added as near misses to show five of each type.

| ID | Type | P(AI) | Possible reason |
|---:|---|---:|---|
| 988 | FP | 0.995 | Polished, balanced film review ("however", structured critique) with low GPT-2 surprisal (2.75) |
| 7887 | FP | 0.993 | Greek text; English-centric features and GPT-2 byte tokens make it look predictable |
| 961 | FP | 0.523 | Very short (95 words) movie review; little evidence either way |
| 1496 | FP (near miss) | 0.443 | Recipe ingredient list: formulaic and highly predictable (surprisal 2.10) |
| 2290 | FP (near miss) | 0.427 | Encyclopedic military history; neutral register resembles AI text |
| 783 | FN | 0.001 | AI recipe imitating scraped human formatting (inline ingredient list, double spaces) |
| 3168 | FN | 0.003 | AI math abstract full of LaTeX; markup dominates the token statistics |
| 6978 | FN | 0.031 | AI paper abstract wrapped in quotation marks, technical vocabulary |
| 8022 | FN | 0.084 | AI Wikipedia-style biography; fluent but with invented facts the model cannot check |
| 3340 | FN | 0.252 | Casual Reddit-style post with typos and U+02BC apostrophes (possible adversarial character swap) |
| 2701 | FN | 0.292 | Portuguese text; high surprisal (5.87) under English GPT-2 looks human |
| 8692 | FN | 0.314 | AI abstract with "Title:/Author:" header and a Greek letter in the title |
| 7361 | FN | 0.428 | AI biography with a "[1]" citation marker and one sentence repeated three times |

## Q4.4 Patterns

Observed:

- Encyclopedic and biographical text (cluster 8) is the weakest domain in validation and LODO, and
  three of eight false negatives are AI biographies written in Wikipedia style.
- Non-English passages fail in both directions (Greek false positive, Portuguese false negative).
  Only 5 validation passages have more than 10% non-ASCII letters, and their AUROC is 0.83.
- Every error lacks a blank line, and short passages have the lowest AUROC.
- Several false negatives copy human document formats: recipe lists, LaTeX, quoted abstracts, citation
  markers.

Speculation (not tested): the language-model features compare a passage with what an English
reference model expects, so any text far from that model (another language, heavy markup) breaks the
signal. Repetition inside a passage (ID 7361) may also be an AI cue our features do not capture.

A fix for one pattern: detect non-English passages and back off to features that do not depend on an
English language model, or add a multilingual reference model. This matters little for Kaggle,
because only about 0.1% of test passages have more than 10% non-ASCII letters, so the
domain-robustness work in Q5 targets the larger shift (fiction and poetry).
