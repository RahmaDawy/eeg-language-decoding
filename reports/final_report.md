# EEG representation learning for brain-signal decoding: ZuCo 1.0 and 2.0

## 1. Question

Can the sentence a person is reading be recovered from their EEG, on subjects not used in training? Does the answer
change with the representation (band power, a self-supervised encoder trained on ZuCo, a frozen pretrained EEG
foundation model), with EEG-quality weighting, with added noise, or when training on one ZuCo dataset and testing on the
other?

## 2. Data and preprocessing

| | ZuCo 1.0 | ZuCo 2.0 |
|---|---|---|
| Subjects used | 12 | 8 (of 18) |
| Recordings (blocks) | 71 | 56 |
| Sentence-reading windows (trigger 10 to 11) | 3125 | 2424 |
| Sentences in the materials | 300 | 349 |
| Windows skipped (shorter than 1 s) | 10 | 60 |
| Usable windows in the representation experiments | 2992 | 1993 |
| Unique sentences among those windows | 264 | 293 |

Natural reading, preprocessed EEG, 500 Hz, 105 channels in both datasets. The flat reference channel Cz was dropped,
leaving 104 common channels. ZuCo 1.0 has 6 blocks per subject (ZPH has no block 2, so 71 instead of 72 recordings);
ZuCo 2.0 has 7 blocks per subject.

- Windows were matched to sentences by order and duration (`scripts/align_windows.py`). Check: the per-subject Spearman
  correlation between window duration and sentence word count is 0.83 (ZuCo 1) and 0.71 (ZuCo 2), against about 0 for
  shuffled sentences.
- Windows shorter than 1 s (500 samples) were skipped because they cannot be real sentence-reading windows: 70 windows,
  median length about 62 samples, maximum 492. By subject: ZDM 3, ZJM 2, ZJS 3, ZMG 2 (ZuCo 1); YAK 42, YDR 10, YFS 4,
  YHS 4 (ZuCo 2). All 42 windows of YAK block 1 were skipped, so that block's trigger structure is probably faulty. The list
  is in `data/processed/skipped_windows.csv`, which is not in the repository. 5479 of 5549 windows have features. After the
  further filters used in the representation experiments (aligned, valid, features, at least one 2.4 s chunk), 4985 windows
  were usable.
- The self-supervised and pretrained encoders use up to four non-overlapping 2.4 s chunks per window at 100 Hz (10716 chunks).

## 3. Methods

**Targets and decoder.** MiniLM sentence embeddings. A ridge decoder maps EEG features to the embedding (features
standardised on training rows; ridge strength chosen on validation subjects' clean data); candidate sentences are ranked by
cosine similarity. Metrics: top-1, recall@5/10, MRR, median rank and sim_margin (similarity to the true sentence minus the
mean similarity to the other candidates), each with its chance value. The same decoder is used for every representation.

**Evaluation.** 4 folds over subjects: test subjects are never in training or validation (3 test subjects per fold in
ZuCo 1, 2 in ZuCo 2). In the `double` regime the test sentences are also unseen (about 15% of the sentences held out for
validation and for test; 39-44 candidates). In `subject_only` all sentences are candidates (about 264-296). `--subject-norm`
standardises each subject with its own statistics; this is transductive and reported separately.

**Controls.** (i) random input: the same pipeline on random numbers of the same dimension, to show that the pipeline does
not produce above-chance scores by itself. (ii) duration-only: reading time as the only feature, a shortcut that EEG
windows carry trivially.

**Representations.**
1. *Band power*: log power in 5 bands for 104 channels (520 features).
2. *Self-supervised encoder*: a small 1D-CNN (3 convolution layers, 128-d output) trained without labels with a contrastive
   (SimCLR-style) loss on 2.4 s chunks at 100 Hz from training subjects only; Gaussian-noise, amplitude-scaling and
   time-shift augmentations; 6 epochs on CPU. A window embedding is the mean over its chunk embeddings. The same network with
   random weights is the "untrained encoder" baseline. Checkpoints: `experiments/checkpoints/`.
3. *Pretrained encoder*: CBraMod (criss-cross transformer, 4.92M parameters, Wang et al., ICLR 2025), checkpoint
   `braindecode/cbramod-pretrained`, **frozen**. CBraMod accepts any number of channels, so all 104 channels were used.
   It expects 200 Hz and windows that are a multiple of 1 s. Adaptation (fixed before looking at results): each 100 Hz chunk
   is upsampled by 2 and cropped to 2.0 s (400 samples, 2 patches); the whole chunk is scaled to a standard deviation of 0.5
   (ZuCo's units were not verified; CBraMod's original preprocessing uses microvolts divided by 100); the encoder output
   (104 channels x 2 patches x 200 features) is averaged over patches and the mean and SD across channels are concatenated
   (400 features); a window embedding is the mean over its chunks. The same architecture with random weights is the
   control for "does pretraining matter".

**Robustness.** Gaussian noise with SD 0.5 / 1 / 2 times the per-channel SD of the chunk or window (mild / moderate /
strong) was added to the **test** EEG only; decoders were trained on clean data. **Quality weighting.** A label-free
artifact score (share of samples beyond 4 robust SDs) was computed per window; during ridge fitting, training windows got weight
1 / (1 + max(0, z)), where z is the artifact score standardised on training rows. It was applied to band power and to the
pretrained features, not to the self-supervised encoder.

**Transfer.** The decoder is fitted on one ZuCo dataset and evaluated on the other (source-only; no target labels or target
EEG were used for fitting), in both directions, with band power only.

**Uncertainty.** For clean EEG in the `double` regime, each test subject's MRR minus chance was computed, and the mean over
subjects was bootstrapped by resampling subjects (5000 resamples, 95% percentile interval). No multiple-comparison correction
and no significance tests were done.

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

**representations, robustness and quality weighting (unseen subjects + unseen sentences; noise added to test EEG only)**

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
| zuco1 | cbramod | frozen_pretrained | clean | 39.0000 | 0.0245 | 0.0256 | 0.1109 | 0.1091 | 0.0033 |
| zuco1 | cbramod | frozen_pretrained | mild | 39.0000 | 0.0268 | 0.0256 | 0.1108 | 0.1091 | 0.0025 |
| zuco1 | cbramod | frozen_pretrained | moderate | 39.0000 | 0.0294 | 0.0256 | 0.1121 | 0.1091 | 0.0015 |
| zuco1 | cbramod | frozen_pretrained | strong | 39.0000 | 0.0177 | 0.0256 | 0.1018 | 0.1091 | -0.0001 |
| zuco1 | cbramod | frozen_pretrained_quality_weighted | clean | 39.0000 | 0.0244 | 0.0256 | 0.1078 | 0.1091 | 0.0033 |
| zuco1 | cbramod | frozen_pretrained_quality_weighted | mild | 39.0000 | 0.0201 | 0.0256 | 0.1072 | 0.1091 | 0.0029 |
| zuco1 | cbramod | frozen_pretrained_quality_weighted | moderate | 39.0000 | 0.0223 | 0.0256 | 0.1067 | 0.1091 | 0.0011 |
| zuco1 | cbramod | frozen_pretrained_quality_weighted | strong | 39.0000 | 0.0221 | 0.0256 | 0.1058 | 0.1091 | -0.0006 |
| zuco1 | cbramod | untrained_cbramod | clean | 39.0000 | 0.0225 | 0.0256 | 0.1022 | 0.1091 | -0.0019 |
| zuco1 | cbramod | random_input | clean | 39.0000 | 0.0223 | 0.0256 | 0.1061 | 0.1091 | -0.0012 |
| zuco2 | cbramod | frozen_pretrained | clean | 40.0000 | 0.0290 | 0.0252 | 0.1173 | 0.1076 | 0.0061 |
| zuco2 | cbramod | frozen_pretrained | mild | 40.0000 | 0.0250 | 0.0252 | 0.1130 | 0.1076 | 0.0089 |
| zuco2 | cbramod | frozen_pretrained | moderate | 40.0000 | 0.0137 | 0.0252 | 0.1024 | 0.1076 | 0.0047 |
| zuco2 | cbramod | frozen_pretrained | strong | 40.0000 | 0.0243 | 0.0252 | 0.1069 | 0.1076 | 0.0015 |
| zuco2 | cbramod | frozen_pretrained_quality_weighted | clean | 40.0000 | 0.0115 | 0.0252 | 0.0896 | 0.1076 | -0.0037 |
| zuco2 | cbramod | frozen_pretrained_quality_weighted | mild | 40.0000 | 0.0103 | 0.0252 | 0.0941 | 0.1076 | -0.0001 |
| zuco2 | cbramod | frozen_pretrained_quality_weighted | moderate | 40.0000 | 0.0122 | 0.0252 | 0.0989 | 0.1076 | 0.0007 |
| zuco2 | cbramod | frozen_pretrained_quality_weighted | strong | 40.0000 | 0.0291 | 0.0252 | 0.1083 | 0.1076 | -0.0012 |
| zuco2 | cbramod | untrained_cbramod | clean | 40.0000 | 0.0199 | 0.0252 | 0.1106 | 0.1076 | 0.0059 |
| zuco2 | cbramod | random_input | clean | 40.0000 | 0.0226 | 0.0252 | 0.1040 | 0.1076 | -0.0023 |

**subject-level analysis (clean EEG, unseen subjects + unseen sentences): mean over test subjects, 95% bootstrap CI from resampling subjects (5000 resamples)**

| dataset | method | n_subjects | mrr_minus_chance_mean | mrr_minus_chance_ci_lo | mrr_minus_chance_ci_hi | sim_margin_mean | sim_margin_ci_lo | sim_margin_ci_hi |
|---|---|---|---|---|---|---|---|---|
| zuco1 | cbramod_frozen | 12 | 0.0027 | -0.0069 | 0.0138 | 0.0035 | -0.0006 | 0.0079 |
| zuco1 | cbramod_untrained | 12 | -0.0068 | -0.0166 | 0.0019 | -0.0020 | -0.0056 | 0.0012 |
| zuco1 | bandpower | 12 | -0.0031 | -0.0128 | 0.0057 | 0.0033 | -0.0029 | 0.0092 |
| zuco1 | random_input | 12 | -0.0000 | -0.0094 | 0.0092 | -0.0012 | -0.0078 | 0.0045 |
| zuco2 | cbramod_frozen | 8 | 0.0101 | 0.0006 | 0.0238 | 0.0067 | -0.0038 | 0.0188 |
| zuco2 | cbramod_untrained | 8 | 0.0029 | -0.0101 | 0.0159 | 0.0062 | 0.0004 | 0.0117 |
| zuco2 | bandpower | 8 | -0.0054 | -0.0193 | 0.0116 | 0.0037 | -0.0027 | 0.0110 |
| zuco2 | random_input | 8 | -0.0127 | -0.0287 | 0.0035 | -0.0050 | -0.0138 | 0.0017 |

<!-- RESULTS:END -->

Figures in `experiments/results/figures/`: `fig_a_eeg_vs_controls.png` (band power vs controls), `fig_b_noise.png`
(performance vs noise, all representations), `fig_c_transfer.png`, `fig_d_subjects.png` (one dot per test subject).

## 5. What worked and what did not

**The pipeline.** The random-input controls are at chance in every experiment, and the window-duration vs word-count
correlation shows that windows are matched to sentences far better than chance. In the `double` regime, no EEG-based fold-mean
MRR differs from chance by more than about 0.02 (the largest gap is -0.018, for quality-weighted CBraMod on ZuCo 2).

**Band power does not retrieve sentences on unseen subjects.** On both datasets, in both regimes, band power is at chance,
with or without subject normalisation. A weak ZuCo 1 hint with subject normalisation (MRR 0.1252 against a chance of 0.1091, sim_margin
0.0138, `double` regime) did not appear in ZuCo 2 (MRR 0.0917, chance 0.0994) and is not separable from reading time.

**The pretrained encoder (CBraMod) did not give a convincing improvement.** On clean EEG the fold-mean MRR was 0.1109 on ZuCo 1
(chance 0.1091) and 0.1173 on ZuCo 2 (chance 0.1076). At subject level, the mean difference from chance was +0.0027 on ZuCo 1
(95% CI -0.0069 to 0.0138, includes zero) and +0.0101 on ZuCo 2 (95% CI 0.0006 to 0.0238, only just excludes zero). The ZuCo 2 mean
depends on one subject: that subject's difference is +0.0545 (see `fig_d_subjects.png`), and the mean of the other seven is +0.0038.
Single high subjects also occur for other methods (ZuCo 2 band power +0.0452 and untrained CBraMod +0.0350; ZuCo 1 frozen CBraMod
+0.0459, with a mean of -0.0012 for the other 11 subjects). On sim_margin the CI of the frozen encoder on ZuCo 2 includes zero
(-0.0038 to 0.0188), while the CI of the randomly initialised copy excludes zero (0.0004 to 0.0117). The two metrics therefore
disagree, the effect is not seen in ZuCo 1, and the untrained copy is not clearly worse. With 16 unadjusted intervals and 8 to 12
subjects, a single borderline interval driven by one subject should not be read as evidence that the pretrained representation
carries sentence information. Band power (ZuCo 1: -0.0031, ZuCo 2: -0.0054) and random input (ZuCo 1: -0.0000, ZuCo 2: -0.0127)
have intervals that include zero.

**Self-supervised learning did not help.** The contrastive loss fell during training (for example ZuCo 1 fold 0: 4.14 to 2.31 over 6 epochs), so
the encoder learned to match augmented views of a chunk, but sentences did not become more retrievable. On clean EEG the trained
encoder's MRR was 0.1065 (ZuCo 1) and 0.0985 (ZuCo 2), against 0.1105 and 0.1003 for the same network with random weights.

**Quality weighting did not help consistently.** For band power, the clean-EEG MRR with weighting versus without was 0.1045
versus 0.1057 (ZuCo 1) and 0.1127 versus 0.1022 (ZuCo 2). For the pretrained features it was 0.1078 versus 0.1109 (ZuCo 1) and
0.0896 versus 0.1173 (ZuCo 2). The sign changes between settings, which is what noise looks like.

**Noise.** Band power and the self-supervised encoder are at chance when clean, so added noise cannot lower them, and several
rows are slightly higher with more noise. That cannot be a real benefit; it measures the size of random fluctuation with 4 folds
and 58 to 115 test windows per fold. In `fig_b_noise.png` the error bars (SD over folds) overlap everywhere and no ordering of
methods holds in both datasets. For frozen CBraMod on ZuCo 2, MRR fell from 0.1173 (clean) to 0.1069 (strong), which is the chance
level (0.1076): the small clean advantage disappeared under noise, while on ZuCo 1 MRR was unchanged up to moderate noise and fell at
strong noise (0.1018).

**Transfer.** In both directions MRR was within about 0.0015 of chance for band power (for example ZuCo 1 to ZuCo 2, unseen sentences: 0.0244
against 0.0237), with and without subject normalisation. Duration-only was slightly above chance (up to about +0.003 MRR). Only reading time transfers.

## 6. Conclusion

In this setup, none of the three representations (band power, a self-supervised encoder trained on ZuCo, a frozen pretrained
EEG foundation model) allowed sentence identity or content to be decoded from EEG for unseen subjects. Quality weighting, noise
robustness and cross-dataset transfer did not change this. One borderline effect for the pretrained encoder on ZuCo 2 is driven by a
single subject, was not reproduced on ZuCo 1, and is not clearly separable from a randomly initialised copy. The only reliably
above-chance signal is reading time. This is a negative result for these methods at this scale; it does not show that ZuCo EEG contains
no sentence information, and a fine-tuned pretrained model or a different adaptation could behave differently.

## 7. Limitations

- Only 8 of the 18 ZuCo 2.0 subjects were used.
- The noise is synthetic, Gaussian, added after recording and only to test data; real artefacts differ.
- Only 4 folds and small test sets (tens of candidate sentences, 8 to 12 test subjects): low power. The bootstrap intervals are
  not corrected for multiple comparisons, and no significance tests were run.
- The window-to-sentence alignment is inferred from order and duration, not from recorded sentence IDs; the correlation check
  shows it is far better than chance, not that it is exact.
- Per-subject normalisation is transductive (it uses the test subject's own statistics).
- 70 short windows were skipped, including all of YAK block 1; ZPH block 2 is absent from the recordings.
- Pretrained model: only CBraMod was run, frozen, without fine-tuning. It was pretrained on clinical EEG (TUEG), and was applied
  to reading EEG after upsampling from 100 Hz, with 2.0 s of each 2.4 s chunk, an assumed amplitude scale, and mean/SD pooling over
  channels that discards the spatial pattern. These choices were made in advance and not tuned; other choices might change the result.
- The self-supervised encoder is small, trained for 6 epochs on CPU on up to a few thousand chunks, and not tuned. Quality weighting was
  not tested on it.
- The self-supervised and pretrained encoders see slightly different inputs (100 Hz, 2.4 s vs 200 Hz, 2.0 s), so they are not a perfectly
  controlled comparison.