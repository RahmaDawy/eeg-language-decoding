# EEG representation learning for brain-signal decoding: ZuCo 1.0 and 2.0

## 1. Question

Can the sentence a person is reading be recovered from their EEG, on subjects not used in training? Does the answer
change with a self-supervised representation, with EEG-quality weighting, with added noise, or when training on one
ZuCo dataset and testing on the other?

## 2. Data and preprocessing

- ZuCo 1.0: 12 subjects. ZuCo 2.0: 8 of 18 subjects. Natural reading, preprocessed EEG, 500 Hz, 105 channels.
  The flat reference channel Cz was dropped (104 channels). ZuCo 1.0 subject ZPH has no block 2 in the recordings used.
- One window per sentence, from trigger 10 to trigger 11: 3125 windows (ZuCo 1) and 2424 (ZuCo 2), 5549 in total.
- Windows were matched to sentences by order and duration (`scripts/align_windows.py`). Check: the per-subject
  Spearman correlation between window duration and sentence word count is 0.83 (ZuCo 1) and 0.71 (ZuCo 2), against
  about 0 for shuffled sentences.
- Windows shorter than 1 s (500 samples) were skipped, because they cannot be real sentence-reading windows. This
  affected 70 windows (median length about 62 samples, maximum 492). All 42 windows of ZuCo 2 subject YAK, block 1, were
  skipped, so that block's trigger structure is probably faulty. The skipped windows are listed in
  `data/processed/skipped_windows.csv`, which is not in the repository. After this and the other filters
  (aligned, valid, at least one 2.4 s chunk), 4985 of 5549 windows were usable in the robustness experiments.

## 3. Methods

**Targets.** MiniLM sentence embeddings. A ridge decoder maps EEG features to the embedding; candidate sentences are
ranked by cosine similarity. Metrics: top-1, recall@5/10, MRR, median rank and sim_margin, each reported with chance.

**Evaluation.** 4 folds over subjects: test subjects are never in training (3 test subjects per fold in ZuCo 1,
2 in ZuCo 2). In the `double` regime the test sentences are also unseen (about 15% of the sentences each held out for
validation and test; 39 and 43 test candidates in ZuCo 1 and 2). In `subject_only` all sentences are candidates.
Ridge strength was chosen on validation subjects' clean data. `--subject-norm` standardises each subject with its
own statistics; this is transductive and is reported separately.

**Controls.** (i) random input: same pipeline on random numbers, to show that the pipeline does not produce
above-chance scores by itself. (ii) duration-only: the sentence's reading time as the only feature, a shortcut that EEG
windows carry trivially (window length).

**Representations.**
1. Band power: log power in 5 bands for 104 channels (520 features).
2. Self-supervised encoder: a small 1D-CNN (3 convolution layers, 128-d output) trained without labels with a
   contrastive (SimCLR-style) loss on 2.4 s chunks at 100 Hz from training subjects only, with Gaussian-noise,
   amplitude-scaling and time-shift augmentations; 6 epochs, CPU. A window embedding is the mean of its chunk
   embeddings. The same network with random weights is the "untrained encoder" baseline.

**Robustness.** Gaussian noise with SD 0.5 / 1 / 2 times the per-channel SD of the window (mild / moderate / strong) was
added to the **test** EEG only; the decoders were trained on clean data. **Quality weighting.** A label-free
artifact score (share of samples beyond 4 robust SDs) was computed per window; during ridge fitting, training windows
got weight 1 / (1 + max(0, z)), where z is the artifact score standardised on training rows. This was only applied to
the band-power decoder.

**Transfer.** The decoder is fitted on one ZuCo dataset and evaluated on the other (source-only), in both directions.

## 4. Results

<!-- RESULTS:START -->
_Means over subject folds. Chance = expected score of a random ranking. No significance tests were run._

**unseen subjects + unseen sentences (raw)**

| dataset | condition | n_candidates | top1 | chance_top1 | recall@5 | chance_recall@5 | mrr | chance_mrr |
|---|---|---|---|---|---|---|---|---|
| zuco1 | eeg_bandpower | 39.0000 | 0.0284 | 0.0256 | 0.1356 | 0.1282 | 0.1110 | 0.1091 |
| zuco1 | duration_only | 39.0000 | 0.0373 | 0.0256 | 0.1493 | 0.1282 | 0.1216 | 0.1091 |
| zuco1 | random_input | 39.0000 | 0.0254 | 0.0256 | 0.1267 | 0.1282 | 0.1079 | 0.1091 |
| zuco2 | eeg_bandpower | 44.0000 | 0.0195 | 0.0227 | 0.1136 | 0.1136 | 0.0957 | 0.0994 |
| zuco2 | duration_only | 44.0000 | 0.0137 | 0.0227 | 0.1055 | 0.1136 | 0.0909 | 0.0994 |
| zuco2 | random_input | 44.0000 | 0.0182 | 0.0227 | 0.1184 | 0.1136 | 0.0982 | 0.0994 |

**unseen subjects + unseen sentences (subject-norm)**

| dataset | condition | n_candidates | top1 | chance_top1 | recall@5 | chance_recall@5 | mrr | chance_mrr | sim_margin |
|---|---|---|---|---|---|---|---|---|---|
| zuco1 | eeg_bandpower | 39.0000 | 0.0352 | 0.0256 | 0.1603 | 0.1282 | 0.1252 | 0.1091 | 0.0138 |
| zuco1 | duration_only | 39.0000 | 0.0373 | 0.0256 | 0.1493 | 0.1282 | 0.1216 | 0.1091 | 0.0034 |
| zuco1 | random_input | 39.0000 | 0.0254 | 0.0256 | 0.1267 | 0.1282 | 0.1079 | 0.1091 | -0.0003 |
| zuco2 | eeg_bandpower | 44.0000 | 0.0138 | 0.0227 | 0.1082 | 0.1136 | 0.0917 | 0.0994 | -0.0020 |
| zuco2 | duration_only | 44.0000 | 0.0137 | 0.0227 | 0.1055 | 0.1136 | 0.0909 | 0.0994 | -0.0009 |
| zuco2 | random_input | 44.0000 | 0.0182 | 0.0227 | 0.1184 | 0.1136 | 0.0982 | 0.0994 | 0.0017 |

**unseen subjects only (raw)**

| dataset | condition | n_candidates | top1 | chance_top1 | recall@5 | chance_recall@5 | mrr | chance_mrr |
|---|---|---|---|---|---|---|---|---|
| zuco1 | eeg_bandpower | 264.0000 | 0.0039 | 0.0038 | 0.0221 | 0.0189 | 0.0254 | 0.0233 |
| zuco1 | duration_only | 264.0000 | 0.0077 | 0.0038 | 0.0254 | 0.0189 | 0.0286 | 0.0233 |
| zuco1 | random_input | 264.0000 | 0.0037 | 0.0038 | 0.0199 | 0.0189 | 0.0235 | 0.0233 |
| zuco2 | eeg_bandpower | 296.2500 | 0.0025 | 0.0034 | 0.0199 | 0.0169 | 0.0219 | 0.0212 |
| zuco2 | duration_only | 296.2500 | 0.0042 | 0.0034 | 0.0233 | 0.0169 | 0.0253 | 0.0212 |
| zuco2 | random_input | 296.2500 | 0.0032 | 0.0034 | 0.0168 | 0.0169 | 0.0210 | 0.0212 |

**unseen subjects only (subject-norm)**

| dataset | condition | n_candidates | top1 | chance_top1 | recall@5 | chance_recall@5 | mrr | chance_mrr | sim_margin |
|---|---|---|---|---|---|---|---|---|---|
| zuco1 | eeg_bandpower | 264.0000 | 0.0048 | 0.0038 | 0.0190 | 0.0189 | 0.0254 | 0.0233 | 0.0077 |
| zuco1 | duration_only | 264.0000 | 0.0077 | 0.0038 | 0.0254 | 0.0189 | 0.0286 | 0.0233 | 0.0082 |
| zuco1 | random_input | 264.0000 | 0.0037 | 0.0038 | 0.0199 | 0.0189 | 0.0235 | 0.0233 | -0.0003 |
| zuco2 | eeg_bandpower | 296.2500 | 0.0059 | 0.0034 | 0.0165 | 0.0169 | 0.0224 | 0.0212 | -0.0011 |
| zuco2 | duration_only | 296.2500 | 0.0042 | 0.0034 | 0.0233 | 0.0169 | 0.0253 | 0.0212 | 0.0094 |
| zuco2 | random_input | 296.2500 | 0.0032 | 0.0034 | 0.0168 | 0.0169 | 0.0210 | 0.0212 | 0.0007 |

**transfer, source-only decoder (raw)**

| source | target | target_subset | condition | n_target_queries | n_candidates | top1 | chance_top1 | mrr | chance_mrr | sim_margin |
|---|---|---|---|---|---|---|---|---|---|---|
| zuco1 | zuco2 | unseen_sentences | eeg_bandpower | 2078.0000 | 259.0000 | 0.0048 | 0.0039 | 0.0244 | 0.0237 | 0.0003 |
| zuco1 | zuco2 | unseen_sentences | duration_only | 2078.0000 | 259.0000 | 0.0034 | 0.0039 | 0.0252 | 0.0237 | 0.0044 |
| zuco1 | zuco2 | unseen_sentences | random_input | 2078.0000 | 259.0000 | 0.0034 | 0.0039 | 0.0229 | 0.0237 | 0.0010 |
| zuco1 | zuco2 | all_sentences | eeg_bandpower | 2364.0000 | 297.0000 | 0.0038 | 0.0034 | 0.0215 | 0.0211 | 0.0016 |
| zuco1 | zuco2 | all_sentences | duration_only | 2364.0000 | 297.0000 | 0.0030 | 0.0034 | 0.0217 | 0.0211 | 0.0045 |
| zuco1 | zuco2 | all_sentences | random_input | 2364.0000 | 297.0000 | 0.0030 | 0.0034 | 0.0209 | 0.0211 | 0.0007 |
| zuco2 | zuco1 | unseen_sentences | eeg_bandpower | 2660.0000 | 226.0000 | 0.0053 | 0.0044 | 0.0266 | 0.0265 | -0.0009 |
| zuco2 | zuco1 | unseen_sentences | duration_only | 2660.0000 | 226.0000 | 0.0045 | 0.0044 | 0.0294 | 0.0265 | 0.0044 |
| zuco2 | zuco1 | unseen_sentences | random_input | 2660.0000 | 226.0000 | 0.0042 | 0.0044 | 0.0263 | 0.0265 | 0.0006 |
| zuco2 | zuco1 | all_sentences | eeg_bandpower | 3115.0000 | 264.0000 | 0.0045 | 0.0038 | 0.0235 | 0.0233 | -0.0005 |
| zuco2 | zuco1 | all_sentences | duration_only | 3115.0000 | 264.0000 | 0.0058 | 0.0038 | 0.0265 | 0.0233 | 0.0049 |
| zuco2 | zuco1 | all_sentences | random_input | 3115.0000 | 264.0000 | 0.0036 | 0.0038 | 0.0236 | 0.0233 | -0.0003 |

**transfer, source-only decoder (subject-norm)**

| source | target | target_subset | condition | n_target_queries | n_candidates | top1 | chance_top1 | mrr | chance_mrr | sim_margin |
|---|---|---|---|---|---|---|---|---|---|---|
| zuco1 | zuco2 | unseen_sentences | eeg_bandpower | 2078.0000 | 259.0000 | 0.0029 | 0.0039 | 0.0225 | 0.0237 | 0.0008 |
| zuco1 | zuco2 | unseen_sentences | duration_only | 2078.0000 | 259.0000 | 0.0034 | 0.0039 | 0.0252 | 0.0237 | 0.0044 |
| zuco1 | zuco2 | unseen_sentences | random_input | 2078.0000 | 259.0000 | 0.0034 | 0.0039 | 0.0229 | 0.0237 | 0.0010 |
| zuco1 | zuco2 | all_sentences | eeg_bandpower | 2364.0000 | 297.0000 | 0.0025 | 0.0034 | 0.0196 | 0.0211 | 0.0003 |
| zuco1 | zuco2 | all_sentences | duration_only | 2364.0000 | 297.0000 | 0.0030 | 0.0034 | 0.0217 | 0.0211 | 0.0045 |
| zuco1 | zuco2 | all_sentences | random_input | 2364.0000 | 297.0000 | 0.0030 | 0.0034 | 0.0209 | 0.0211 | 0.0007 |
| zuco2 | zuco1 | unseen_sentences | eeg_bandpower | 2660.0000 | 226.0000 | 0.0049 | 0.0044 | 0.0267 | 0.0265 | -0.0020 |
| zuco2 | zuco1 | unseen_sentences | duration_only | 2660.0000 | 226.0000 | 0.0045 | 0.0044 | 0.0294 | 0.0265 | 0.0044 |
| zuco2 | zuco1 | unseen_sentences | random_input | 2660.0000 | 226.0000 | 0.0042 | 0.0044 | 0.0263 | 0.0265 | 0.0006 |
| zuco2 | zuco1 | all_sentences | eeg_bandpower | 3115.0000 | 264.0000 | 0.0045 | 0.0038 | 0.0233 | 0.0233 | -0.0006 |
| zuco2 | zuco1 | all_sentences | duration_only | 3115.0000 | 264.0000 | 0.0058 | 0.0038 | 0.0265 | 0.0233 | 0.0049 |
| zuco2 | zuco1 | all_sentences | random_input | 3115.0000 | 264.0000 | 0.0036 | 0.0038 | 0.0236 | 0.0233 | -0.0003 |

**robustness and self-supervised encoder (unseen subjects + unseen sentences; noise added to test EEG only)**

| dataset | representation | method | noise | n_candidates | top1 | chance_top1 | mrr | chance_mrr | sim_margin |
|---|---|---|---|---|---|---|---|---|---|
| zuco1 | bandpower | plain | clean | 39.0000 | 0.0200 | 0.0256 | 0.1057 | 0.1091 | 0.0031 |
| zuco1 | bandpower | plain | mild | 39.0000 | 0.0156 | 0.0256 | 0.1006 | 0.1091 | 0.0014 |
| zuco1 | bandpower | plain | moderate | 39.0000 | 0.0316 | 0.0256 | 0.1130 | 0.1091 | 0.0002 |
| zuco1 | bandpower | plain | strong | 39.0000 | 0.0244 | 0.0256 | 0.1028 | 0.1091 | -0.0010 |
| zuco1 | bandpower | quality_weighted | clean | 39.0000 | 0.0179 | 0.0256 | 0.1045 | 0.1091 | 0.0022 |
| zuco1 | bandpower | quality_weighted | mild | 39.0000 | 0.0249 | 0.0256 | 0.1044 | 0.1091 | 0.0001 |
| zuco1 | bandpower | quality_weighted | moderate | 39.0000 | 0.0227 | 0.0256 | 0.1049 | 0.1091 | -0.0002 |
| zuco1 | bandpower | quality_weighted | strong | 39.0000 | 0.0268 | 0.0256 | 0.1053 | 0.1091 | -0.0012 |
| zuco1 | bandpower | random_input | clean | 39.0000 | 0.0209 | 0.0256 | 0.1003 | 0.1091 | -0.0024 |
| zuco1 | ssl | ssl_trained | clean | 39.0000 | 0.0243 | 0.0256 | 0.1065 | 0.1091 | 0.0035 |
| zuco1 | ssl | ssl_trained | mild | 39.0000 | 0.0221 | 0.0256 | 0.1047 | 0.1091 | 0.0034 |
| zuco1 | ssl | ssl_trained | moderate | 39.0000 | 0.0220 | 0.0256 | 0.1044 | 0.1091 | 0.0034 |
| zuco1 | ssl | ssl_trained | strong | 39.0000 | 0.0197 | 0.0256 | 0.1015 | 0.1091 | 0.0011 |
| zuco1 | ssl | untrained_encoder | clean | 39.0000 | 0.0293 | 0.0256 | 0.1105 | 0.1091 | 0.0056 |
| zuco1 | ssl | untrained_encoder | mild | 39.0000 | 0.0338 | 0.0256 | 0.1152 | 0.1091 | 0.0064 |
| zuco1 | ssl | untrained_encoder | moderate | 39.0000 | 0.0293 | 0.0256 | 0.1166 | 0.1091 | 0.0041 |
| zuco1 | ssl | untrained_encoder | strong | 39.0000 | 0.0269 | 0.0256 | 0.1144 | 0.1091 | 0.0017 |
| zuco1 | ssl | random_input | clean | 39.0000 | 0.0268 | 0.0256 | 0.1094 | 0.1091 | 0.0002 |
| zuco2 | bandpower | plain | clean | 40.0000 | 0.0170 | 0.0252 | 0.1022 | 0.1076 | 0.0036 |
| zuco2 | bandpower | plain | mild | 40.0000 | 0.0170 | 0.0252 | 0.1068 | 0.1076 | 0.0042 |
| zuco2 | bandpower | plain | moderate | 40.0000 | 0.0341 | 0.0252 | 0.1181 | 0.1076 | 0.0083 |
| zuco2 | bandpower | plain | strong | 40.0000 | 0.0182 | 0.0252 | 0.1110 | 0.1076 | 0.0047 |
| zuco2 | bandpower | quality_weighted | clean | 40.0000 | 0.0298 | 0.0252 | 0.1127 | 0.1076 | 0.0033 |
| zuco2 | bandpower | quality_weighted | mild | 40.0000 | 0.0262 | 0.0252 | 0.1111 | 0.1076 | 0.0032 |
| zuco2 | bandpower | quality_weighted | moderate | 40.0000 | 0.0327 | 0.0252 | 0.1169 | 0.1076 | 0.0039 |
| zuco2 | bandpower | quality_weighted | strong | 40.0000 | 0.0343 | 0.0252 | 0.1192 | 0.1076 | 0.0030 |
| zuco2 | bandpower | random_input | clean | 40.0000 | 0.0270 | 0.0252 | 0.1127 | 0.1076 | 0.0031 |
| zuco2 | ssl | ssl_trained | clean | 40.0000 | 0.0235 | 0.0252 | 0.0985 | 0.1076 | -0.0045 |
| zuco2 | ssl | ssl_trained | mild | 40.0000 | 0.0271 | 0.0252 | 0.1007 | 0.1076 | -0.0047 |
| zuco2 | ssl | ssl_trained | moderate | 40.0000 | 0.0293 | 0.0252 | 0.1097 | 0.1076 | -0.0020 |
| zuco2 | ssl | ssl_trained | strong | 40.0000 | 0.0394 | 0.0252 | 0.1187 | 0.1076 | 0.0025 |
| zuco2 | ssl | untrained_encoder | clean | 40.0000 | 0.0204 | 0.0252 | 0.1003 | 0.1076 | -0.0034 |
| zuco2 | ssl | untrained_encoder | mild | 40.0000 | 0.0266 | 0.0252 | 0.1096 | 0.1076 | -0.0009 |
| zuco2 | ssl | untrained_encoder | moderate | 40.0000 | 0.0247 | 0.0252 | 0.1030 | 0.1076 | -0.0042 |
| zuco2 | ssl | untrained_encoder | strong | 40.0000 | 0.0319 | 0.0252 | 0.1139 | 0.1076 | 0.0017 |
| zuco2 | ssl | random_input | clean | 40.0000 | 0.0266 | 0.0252 | 0.1067 | 0.1076 | -0.0014 |

<!-- RESULTS:END -->

Figures: `fig_a_eeg_vs_controls.png`, `fig_b_noise.png`, `fig_c_transfer.png` in `experiments/results/figures/`.

## 5. What worked and what did not

**Pipeline.** The random-input control is at chance in every experiment, and the window-duration vs word-count
correlation shows that windows are matched to sentences far better than chance. The negative results below are
therefore not an artefact of a broken evaluation, although a more sensitive decoder could still find something this one misses.

**EEG band power does not retrieve sentences on unseen subjects.** On both datasets, in both regimes, band power
is at chance level, with or without subject normalisation. In the robustness runs (unseen subjects and sentences) every
MRR differs from chance by about 0.012 or less, and the random-input controls fluctuate by a similar amount (see the
noise table above).

**Reading time is the only signal that is above chance.** The duration-only control is slightly above chance, and
it is the only thing that transfers between datasets. A weak ZuCo 1 hint (sim_margin 0.014, double regime) did not
replicate in ZuCo 2 and cannot be separated from reading time.

**Self-supervised learning did not help.** The contrastive loss fell during training (for example, in the ZuCo 1
fold 0, from 4.14 to 2.31 over 6 epochs), so the encoder learned to match augmented views of the same chunk. That did not
make sentences more retrievable: the trained encoder is at chance and is not better than the same network with
random weights (see the SSL rows in the noise table).

**Quality weighting did not help.** Weighted and plain band-power decoders are indistinguishable at every noise level.

**Noise did not matter, because there was nothing to lose.** Clean band power is already at chance, so added noise
cannot lower it. Some rows are slightly higher with more noise (for example ZuCo 2, trained encoder, strong
noise has a higher top-1 than clean). Added noise cannot plausibly help, so this is a measure of the size of
random fluctuation with 4 folds and 58 to 115 test windows per fold.

**Transfer.** Cross-dataset transfer is at chance in both directions; only reading time transfers.

## 6. Conclusion

In this setup, with 104-channel band-power features and a small self-supervised encoder, no information about
sentence identity or content could be decoded from EEG for unseen subjects, and nothing changed with quality weighting,
added noise or training on the other dataset. The only above-chance signal is reading time. This is a negative result
for these methods at this scale, not a proof that ZuCo EEG contains no sentence information.

## 7. Limitations

- Only 8 of the 18 ZuCo 2.0 subjects were used.
- The noise is synthetic, Gaussian, added after recording and only to the test data; real artefacts differ.
- Only 4 folds, with small test sets (tens of candidate sentences): the analysis has low power, and no
  significance tests were run, so small differences between rows should not be interpreted.
- The window-to-sentence alignment is inferred from order and duration, not from recorded sentence IDs; the
  correlation check shows that it is far better than chance, not that it is exact.
- Per-subject normalisation is transductive (it uses statistics of the test subject's own recording).
- 70 short windows were skipped, including all of YAK block 1; ZPH block 2 is absent from the recordings.
- No pretrained EEG representation model was run: no public model was verified to work with these 104 channels
  at this setup, and I did not want to report an unverified adaptation. The comparison of "two representation
  approaches" is therefore band power vs a self-supervised encoder trained on ZuCo (plus its untrained version), not a
  comparison with a pretrained EEG foundation model.
- The self-supervised encoder is small, was trained for 6 epochs on CPU on up to a few thousand chunks, and was not tuned.
  Quality weighting was only tested for band power.