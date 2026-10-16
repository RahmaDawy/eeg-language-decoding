# EEG Representation Learning for Brain Signal Decoding (ZuCo 1.0 + 2.0)

Student research project. Question: can information about the sentence a person is reading be recovered from
their EEG, for **subjects the model has never seen**, and does this survive added noise and a change of dataset?

**Short answer (details, numbers and limitations below): in this setup, no.** EEG band power and a small
self-supervised encoder both perform at chance level for sentence retrieval on unseen subjects. Random-input
controls are also at chance (so the pipeline itself is valid). Only a reading-time shortcut (sentence duration)
is slightly above chance, and it is the only thing that transfers between ZuCo 1 and ZuCo 2.

## What was done

- **Data**: ZuCo 1.0 (12 subjects) and the first 8 of the 18 ZuCo 2.0 subjects that were downloaded, natural reading.
  Preprocessed EEG, 500 Hz, 105 channels; the flat reference channel Cz is dropped (104 channels).
- **Windows**: one window per sentence, from trigger 10 to trigger 11 (3125 windows in ZuCo 1, 2424 in ZuCo 2).
  Windows are matched to sentences by order and duration (`scripts/align_windows.py`). Sanity check: per-subject
  Spearman correlation between window duration and sentence word count is 0.83 (ZuCo 1) and 0.71 (ZuCo 2), and
  about 0 when sentences are shuffled.
- **Decoding task**: predict the MiniLM sentence embedding from EEG with a ridge decoder, then rank candidate
  sentences by cosine similarity. Metrics: top-1, recall@5/10, MRR, median rank, sim_margin (similarity to the true
  sentence minus the mean similarity to the others), each with its chance value.
- **Evaluation**: 4 folds over subjects (test subjects never seen in training). `double` regime = unseen subjects
  AND unseen sentences; `subject_only` = unseen subjects only.
- **Controls**: random input (pipeline check) and duration-only (reading-time shortcut).
- **Representations**: (1) log band power, 104 channels x 5 bands; (2) a small 1D-CNN encoder trained without labels
  (contrastive, SimCLR-style) on training-subject chunks, compared with the same network untrained.
- **Robustness**: Gaussian noise of 0.5 / 1 / 2 x the per-channel SD added to the **test** EEG; a label-free
  artifact score used to down-weight noisy training windows (quality-weighted ridge).
- **Transfer**: train on one ZuCo dataset, test on the other (source-only decoder).

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

All tables, including the noise/SSL runs: `experiments/results/results_summary.md`.
<!-- RESULTS:END -->

Figures: `experiments/results/figures/` (`fig_a_eeg_vs_controls.png`, `fig_b_noise.png`, `fig_c_transfer.png`).
Full discussion: `reports/final_report.md`.

## Limitations

- Only 8 of the 18 ZuCo 2.0 subjects are used.
- Noise is synthetic Gaussian noise added after recording, to test data only.
- Only 4 subject folds; test sets are small (tens of sentences), so small differences are not interpretable.
- Window-to-sentence alignment is inferred from order and duration, not from recorded sentence IDs.
- Per-subject normalisation (`--subject-norm`) is transductive: it uses the test subject's own data statistics.
- 70 of 5549 windows (shorter than 1 s) were skipped; one ZuCo 2 block (YAK block 1) lost all its windows.
- No pretrained EEG foundation model was run: no suitable public model was verified for this 104-channel setup.
  The "pretrained representation" comparison is replaced by band power vs a self-supervised encoder trained on ZuCo.
- The self-supervised encoder is small and was trained for 6 epochs on CPU; it was not tuned.

## Setup (Windows, PowerShell)

```
cd C:\eeg-language-decoding
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install matplotlib pytest
pytest -q
```

Optional (Colab, not tested by the author): `!git clone https://github.com/RahmaDawy/eeg-language-decoding`,
then `%cd eeg-language-decoding`, `!pip install -e . matplotlib pytest`, copy the data into `data/raw/zuco1` and
`data/raw/zuco2` (for example from Google Drive), and run the same commands as below with `!` in front.

## Data access and licence

The data is **not** in this repository. Download the preprocessed natural-reading recordings from OSF:

- ZuCo 1.0: https://osf.io/q3zws/overview
- ZuCo 2.0: https://osf.io/2urht/overview

Put them in `data/raw/zuco1` and `data/raw/zuco2`. Use of the data is governed by the terms on those OSF pages;
check them before redistributing anything. `data/` and all `.mat` files are git-ignored.

## Reproduce

```
python scripts\align_windows.py
```
(step 1b: run the scripts that create `data\processed\sentences.csv`, `sent_emb.npy` and `features.npy`)
```
python scripts\run_experiments.py --regime double
python scripts\run_experiments.py --regime double --subject-norm
python scripts\run_experiments.py --regime subject_only
python scripts\run_experiments.py --regime subject_only --subject-norm
python scripts\run_transfer.py
python scripts\run_transfer.py --subject-norm
python scripts\make_signals.py
python scripts\run_robust_ssl.py
python scripts\make_figures.py
```

`make_signals.py` took about 10 min and `run_robust_ssl.py` about 3 min on a CPU laptop.