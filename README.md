# EEG Representation Learning for Brain Signal Decoding (ZuCo 1.0 + 2.0)

Student research project. Question: can information about the sentence a person is reading be recovered from
their EEG, **for subjects the model has never seen**, and does this survive added noise and a change of dataset?
Three EEG representations are compared: log band power, a small self-supervised (contrastive) encoder trained on
ZuCo, and a **frozen pretrained EEG foundation model (CBraMod)**.

**Short answer (numbers and limitations below): no reliable evidence.** Band power, the self-supervised encoder
and the pretrained encoder all perform at or very close to chance for sentence retrieval on unseen subjects. Random-input
controls are also at chance, so the pipeline itself is valid. The one slightly above-chance result is a borderline
effect for the pretrained encoder on ZuCo 2 that is driven by a single subject, does not appear in ZuCo 1, and is not
clearly different from a randomly initialised copy of the same network. Reading time (sentence duration) is the only
thing that transfers between datasets.

## What was done

- **Data**: ZuCo 1.0 (12 subjects, 71 recordings) and 8 of the 18 ZuCo 2.0 subjects (56 recordings), natural reading,
  preprocessed EEG, 500 Hz, 105 channels; the flat reference channel Cz is dropped (104 channels).
- **Windows**: one window per sentence, trigger 10 to trigger 11 (3125 in ZuCo 1, 2424 in ZuCo 2), matched to sentences
  by order and duration (`scripts/align_windows.py`). Check: the per-subject Spearman correlation between window duration
  and sentence word count is 0.83 (ZuCo 1) and 0.71 (ZuCo 2), and about 0 for shuffled sentences.
  70 windows shorter than 1 s were skipped (`scripts/make_signals.py`).
- **Decoding**: ridge regression from EEG features to the MiniLM sentence embedding, then ranking of candidate sentences
  by cosine similarity. Metrics: top-1, recall@5/10, MRR, median rank, sim_margin, each with its chance value.
- **Evaluation**: 4 folds over subjects (test subjects never seen in training or validation). `double` regime: unseen
  subjects AND unseen sentences (about 39-44 candidates). `subject_only`: unseen subjects only (about 264-296 candidates).
- **Representations**: (1) log band power, 104 channels x 5 bands; (2) a small 1D-CNN trained without labels
  (SimCLR-style contrastive loss) on training-subject chunks, compared with the same network untrained;
  (3) frozen **CBraMod** (`braindecode/cbramod-pretrained`), compared with the same architecture with random weights.
- **Controls**: random input (same decoder and split), duration-only (reading-time shortcut).
- **Robustness**: Gaussian noise of 0.5 / 1 / 2 times the per-channel SD added to the **test** EEG only; a label-free
  artifact score used to down-weight noisy training windows (quality weighting).
- **Transfer**: decoder fitted on one ZuCo dataset, tested on the other (source-only), both directions, band power only.
- **Uncertainty**: subject-level bootstrap (resampling test subjects) for the clean-EEG comparison.

## Results

<!-- RESULTS:START -->
_Means over folds; chance is the expected score of a random ranking._

| experiment | dataset | condition | top1 | chance_top1 | mrr | chance_mrr |
|---|---|---|---|---|---|---|
| double, raw | zuco1 | eeg_bandpower | 0.0284 | 0.0256 | 0.1110 | 0.1091 |
| double, raw | zuco1 | duration_only | 0.0373 | 0.0256 | 0.1216 | 0.1091 |
| double, raw | zuco1 | random_input | 0.0254 | 0.0256 | 0.1079 | 0.1091 |
| double, raw | zuco2 | eeg_bandpower | 0.0195 | 0.0227 | 0.0957 | 0.0994 |
| double, raw | zuco2 | duration_only | 0.0137 | 0.0227 | 0.0909 | 0.0994 |
| double, raw | zuco2 | random_input | 0.0182 | 0.0227 | 0.0982 | 0.0994 |
| double, subject-norm | zuco1 | eeg_bandpower | 0.0352 | 0.0256 | 0.1252 | 0.1091 |
| double, subject-norm | zuco1 | duration_only | 0.0373 | 0.0256 | 0.1216 | 0.1091 |
| double, subject-norm | zuco1 | random_input | 0.0254 | 0.0256 | 0.1079 | 0.1091 |
| double, subject-norm | zuco2 | eeg_bandpower | 0.0138 | 0.0227 | 0.0917 | 0.0994 |
| double, subject-norm | zuco2 | duration_only | 0.0137 | 0.0227 | 0.0909 | 0.0994 |
| double, subject-norm | zuco2 | random_input | 0.0182 | 0.0227 | 0.0982 | 0.0994 |
| subject_only, raw | zuco1 | eeg_bandpower | 0.0039 | 0.0038 | 0.0254 | 0.0233 |
| subject_only, raw | zuco1 | duration_only | 0.0077 | 0.0038 | 0.0286 | 0.0233 |
| subject_only, raw | zuco1 | random_input | 0.0037 | 0.0038 | 0.0235 | 0.0233 |
| subject_only, raw | zuco2 | eeg_bandpower | 0.0025 | 0.0034 | 0.0219 | 0.0212 |
| subject_only, raw | zuco2 | duration_only | 0.0042 | 0.0034 | 0.0253 | 0.0212 |
| subject_only, raw | zuco2 | random_input | 0.0032 | 0.0034 | 0.0210 | 0.0212 |
| subject_only, subject-norm | zuco1 | eeg_bandpower | 0.0048 | 0.0038 | 0.0254 | 0.0233 |
| subject_only, subject-norm | zuco1 | duration_only | 0.0077 | 0.0038 | 0.0286 | 0.0233 |
| subject_only, subject-norm | zuco1 | random_input | 0.0037 | 0.0038 | 0.0235 | 0.0233 |
| subject_only, subject-norm | zuco2 | eeg_bandpower | 0.0059 | 0.0034 | 0.0224 | 0.0212 |
| subject_only, subject-norm | zuco2 | duration_only | 0.0042 | 0.0034 | 0.0253 | 0.0212 |
| subject_only, subject-norm | zuco2 | random_input | 0.0032 | 0.0034 | 0.0210 | 0.0212 |
| transfer (raw, unseen_sentences) | zuco1->zuco2 | eeg_bandpower | 0.0048 | 0.0039 | 0.0244 | 0.0237 |
| transfer (raw, unseen_sentences) | zuco1->zuco2 | duration_only | 0.0034 | 0.0039 | 0.0252 | 0.0237 |
| transfer (raw, unseen_sentences) | zuco1->zuco2 | random_input | 0.0034 | 0.0039 | 0.0229 | 0.0237 |
| transfer (raw, all_sentences) | zuco1->zuco2 | eeg_bandpower | 0.0038 | 0.0034 | 0.0215 | 0.0211 |
| transfer (raw, all_sentences) | zuco1->zuco2 | duration_only | 0.0030 | 0.0034 | 0.0217 | 0.0211 |
| transfer (raw, all_sentences) | zuco1->zuco2 | random_input | 0.0030 | 0.0034 | 0.0209 | 0.0211 |
| transfer (raw, unseen_sentences) | zuco2->zuco1 | eeg_bandpower | 0.0053 | 0.0044 | 0.0266 | 0.0265 |
| transfer (raw, unseen_sentences) | zuco2->zuco1 | duration_only | 0.0045 | 0.0044 | 0.0294 | 0.0265 |
| transfer (raw, unseen_sentences) | zuco2->zuco1 | random_input | 0.0042 | 0.0044 | 0.0263 | 0.0265 |
| transfer (raw, all_sentences) | zuco2->zuco1 | eeg_bandpower | 0.0045 | 0.0038 | 0.0235 | 0.0233 |
| transfer (raw, all_sentences) | zuco2->zuco1 | duration_only | 0.0058 | 0.0038 | 0.0265 | 0.0233 |
| transfer (raw, all_sentences) | zuco2->zuco1 | random_input | 0.0036 | 0.0038 | 0.0236 | 0.0233 |
| transfer (subject-norm, unseen_sentences) | zuco1->zuco2 | eeg_bandpower | 0.0029 | 0.0039 | 0.0225 | 0.0237 |
| transfer (subject-norm, unseen_sentences) | zuco1->zuco2 | duration_only | 0.0034 | 0.0039 | 0.0252 | 0.0237 |
| transfer (subject-norm, unseen_sentences) | zuco1->zuco2 | random_input | 0.0034 | 0.0039 | 0.0229 | 0.0237 |
| transfer (subject-norm, all_sentences) | zuco1->zuco2 | eeg_bandpower | 0.0025 | 0.0034 | 0.0196 | 0.0211 |
| transfer (subject-norm, all_sentences) | zuco1->zuco2 | duration_only | 0.0030 | 0.0034 | 0.0217 | 0.0211 |
| transfer (subject-norm, all_sentences) | zuco1->zuco2 | random_input | 0.0030 | 0.0034 | 0.0209 | 0.0211 |
| transfer (subject-norm, unseen_sentences) | zuco2->zuco1 | eeg_bandpower | 0.0049 | 0.0044 | 0.0267 | 0.0265 |
| transfer (subject-norm, unseen_sentences) | zuco2->zuco1 | duration_only | 0.0045 | 0.0044 | 0.0294 | 0.0265 |
| transfer (subject-norm, unseen_sentences) | zuco2->zuco1 | random_input | 0.0042 | 0.0044 | 0.0263 | 0.0265 |
| transfer (subject-norm, all_sentences) | zuco2->zuco1 | eeg_bandpower | 0.0045 | 0.0038 | 0.0233 | 0.0233 |
| transfer (subject-norm, all_sentences) | zuco2->zuco1 | duration_only | 0.0058 | 0.0038 | 0.0265 | 0.0233 |
| transfer (subject-norm, all_sentences) | zuco2->zuco1 | random_input | 0.0036 | 0.0038 | 0.0236 | 0.0233 |
| representations, clean EEG, double | zuco1 | bandpower / plain | 0.0200 | 0.0256 | 0.1057 | 0.1091 |
| representations, clean EEG, double | zuco1 | bandpower / quality_weighted | 0.0179 | 0.0256 | 0.1045 | 0.1091 |
| representations, clean EEG, double | zuco1 | bandpower / random_input | 0.0209 | 0.0256 | 0.1003 | 0.1091 |
| representations, clean EEG, double | zuco1 | ssl / ssl_trained | 0.0243 | 0.0256 | 0.1065 | 0.1091 |
| representations, clean EEG, double | zuco1 | ssl / untrained_encoder | 0.0293 | 0.0256 | 0.1105 | 0.1091 |
| representations, clean EEG, double | zuco1 | ssl / random_input | 0.0268 | 0.0256 | 0.1094 | 0.1091 |
| representations, clean EEG, double | zuco2 | bandpower / plain | 0.0170 | 0.0252 | 0.1022 | 0.1076 |
| representations, clean EEG, double | zuco2 | bandpower / quality_weighted | 0.0298 | 0.0252 | 0.1127 | 0.1076 |
| representations, clean EEG, double | zuco2 | bandpower / random_input | 0.0270 | 0.0252 | 0.1127 | 0.1076 |
| representations, clean EEG, double | zuco2 | ssl / ssl_trained | 0.0235 | 0.0252 | 0.0985 | 0.1076 |
| representations, clean EEG, double | zuco2 | ssl / untrained_encoder | 0.0204 | 0.0252 | 0.1003 | 0.1076 |
| representations, clean EEG, double | zuco2 | ssl / random_input | 0.0266 | 0.0252 | 0.1067 | 0.1076 |
| representations, clean EEG, double | zuco1 | cbramod / frozen_pretrained | 0.0245 | 0.0256 | 0.1109 | 0.1091 |
| representations, clean EEG, double | zuco1 | cbramod / frozen_pretrained_quality_weighted | 0.0244 | 0.0256 | 0.1078 | 0.1091 |
| representations, clean EEG, double | zuco1 | cbramod / untrained_cbramod | 0.0225 | 0.0256 | 0.1022 | 0.1091 |
| representations, clean EEG, double | zuco1 | cbramod / random_input | 0.0223 | 0.0256 | 0.1061 | 0.1091 |
| representations, clean EEG, double | zuco2 | cbramod / frozen_pretrained | 0.0290 | 0.0252 | 0.1173 | 0.1076 |
| representations, clean EEG, double | zuco2 | cbramod / frozen_pretrained_quality_weighted | 0.0115 | 0.0252 | 0.0896 | 0.1076 |
| representations, clean EEG, double | zuco2 | cbramod / untrained_cbramod | 0.0199 | 0.0252 | 0.1106 | 0.1076 |
| representations, clean EEG, double | zuco2 | cbramod / random_input | 0.0226 | 0.0252 | 0.1040 | 0.1076 |

_Subject-level (clean EEG, double regime): mean over test subjects of MRR minus chance, 95% bootstrap CI over subjects._

| dataset | method | n_subjects | mrr_minus_chance_mean | mrr_minus_chance_ci_lo | mrr_minus_chance_ci_hi |
|---|---|---|---|---|---|
| zuco1 | cbramod_frozen | 12 | 0.0027 | -0.0069 | 0.0138 |
| zuco1 | cbramod_untrained | 12 | -0.0068 | -0.0166 | 0.0019 |
| zuco1 | bandpower | 12 | -0.0031 | -0.0128 | 0.0057 |
| zuco1 | random_input | 12 | -0.0000 | -0.0094 | 0.0092 |
| zuco2 | cbramod_frozen | 8 | 0.0101 | 0.0006 | 0.0238 |
| zuco2 | cbramod_untrained | 8 | 0.0029 | -0.0101 | 0.0159 |
| zuco2 | bandpower | 8 | -0.0054 | -0.0193 | 0.0116 |
| zuco2 | random_input | 8 | -0.0127 | -0.0287 | 0.0035 |

All tables: `experiments/results/results_summary.md`.
<!-- RESULTS:END -->

Figures in `experiments/results/figures/`: `fig_a_eeg_vs_controls.png`, `fig_b_noise.png`, `fig_c_transfer.png`,
`fig_d_subjects.png`. Discussion: `reports/final_report.md`. Trained self-supervised encoders:
`experiments/checkpoints/` (one per dataset and fold).

## Limitations

- Only 8 of the 18 ZuCo 2.0 subjects are used.
- Noise is synthetic Gaussian noise added after recording, to test data only.
- Only 4 subject folds; test sets are small (tens of candidate sentences, 8 to 12 test subjects).
  Differences of about 0.01 MRR are within fold-to-fold variation, and single subjects can move a group mean by that
  much. No multiple-comparison correction was applied to the bootstrap intervals.
- Window-to-sentence alignment is inferred from order and duration, not from recorded sentence IDs.
- Per-subject normalisation (`--subject-norm`) is transductive: it uses the test subject's own statistics.
- The pretrained encoder (CBraMod) was pretrained on clinical EEG (TUEG); it was used frozen, without fine-tuning,
  and with several fixed adaptations (upsampling from 100 Hz, 2.0 s windows, assumed amplitude scale, mean/SD pooling
  over channels). Other adaptations or fine-tuning might behave differently. Only one pretrained model was run.
- The self-supervised encoder is small and was trained for 6 epochs on CPU; it was not tuned.

## Setup (Windows, PowerShell)

```
cd C:\eeg-language-decoding
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install matplotlib pytest "braindecode[hub]" osfclient
pytest -q
```

The first run of `scripts/run_pretrained.py` downloads the CBraMod checkpoint (about 20 MB) from Hugging Face.
Optional Colab (not tested by the author): `!git clone https://github.com/RahmaDawy/eeg-language-decoding`,
`%cd eeg-language-decoding`, `!pip install -e . matplotlib pytest "braindecode[hub]" osfclient`, put the data in
`data/raw/zuco1` and `data/raw/zuco2` (for example from Google Drive), then run the commands below with `!` in front.

## Data access and licence

The data is **not** in this repository. Download the preprocessed natural-reading recordings and the task materials
from OSF:

- ZuCo 1.0: https://osf.io/q3zws/overview (OSF project `q3zws`)
- ZuCo 2.0: https://osf.io/2urht/overview (OSF project `2urht`)

`scripts/download_zuco.py` downloads selected files into `data/raw`. Example for ZuCo 2 (add `--dry-run` first to list
what would be downloaded; for ZuCo 1 use `--project q3zws`, `--out data/raw/zuco1` and the task folder name shown on OSF):

```
python scripts\download_zuco.py --project 2urht --task "task1 - NR" --out data/raw/zuco2 --areas Preprocessed task_materials --subjects YAC YAG YAK YDG YDR YFR YFS YHS
```

Do not run other scripts with `--help`: they do not parse arguments and would run in full.
Use of the data is governed by the terms on the OSF pages; check them before redistributing anything. `data/` and all
`.mat` and `.npy` files are git-ignored. The CBraMod checkpoint is distributed by braindecode (BSD-3-Clause for the
code); its pretraining data (TUEG) has its own terms, so check the Hugging Face model page before reuse.

## Reproduce

Preparation. `build_dataset.py` takes no arguments and processes everything in `data/raw` (about a minute or two).
Its final summary section is outdated: it prints "aligned 0", a stale hint to run `run_baseline.py`, and ends with a
`KeyError` traceback. This happens after all its output files (`windows.csv`, `features.npy`, `sentences.csv`,
`sent_emb.npy`) have been saved, so it can be ignored. `align_windows.py` then creates `windows_aligned.csv`.
```
python scripts\build_dataset.py
python scripts\align_windows.py
```
Band-power experiments with controls (unseen subjects; transfer):
```
python scripts\run_experiments.py --regime double
python scripts\run_experiments.py --regime double --subject-norm
python scripts\run_experiments.py --regime subject_only
python scripts\run_experiments.py --regime subject_only --subject-norm
python scripts\run_transfer.py
python scripts\run_transfer.py --subject-norm
```
Noise, quality weighting, self-supervised and pretrained encoders, figures and tests:
```
python scripts\make_signals.py
python scripts\run_robust_ssl.py
python scripts\run_pretrained.py
python scripts\make_figures.py
pytest -q
```
Approximate run times on a CPU laptop: `make_signals.py` 10 min, `run_robust_ssl.py` 3 min, `run_pretrained.py` 31 min
(encodings are cached in `data/processed/cbramod_*.npy`), the rest a few minutes. `--pilot` on `run_robust_ssl.py` and
`run_pretrained.py` is a quick smoke test; its outputs are not results.