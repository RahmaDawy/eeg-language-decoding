# Final report: EEG representations and sentence information in ZuCo

## Question
Can EEG recorded while people read sentences identify the sentence they read, and does this hold for unseen subjects, noisy signals, and a second dataset?

## Data
- **ZuCo 1.0:** 12 subjects, natural reading, 105 EEG channels at 500 Hz. 71 preprocessed block recordings (subject ZPH is missing block 2). The "Matlab files" give 300 sentences per subject with the true sentence, raw per-sentence EEG, the authors' band features, and per-word features.
- **ZuCo 2.0:** 8 of 18 subjects (56 block recordings, same 105 channels and 500 Hz).
- The channel labels and their order are identical in both datasets; the reference channel Cz is all zeros and was dropped (104 channels).
- Recordings were already preprocessed by the dataset authors; we did not re-filter.

## Methods
- **Units:** sentence windows from triggers 10 -> 11 (ZuCo 1.0: 3,125 windows; ZuCo 2.0: 2,424), aligned to sentences by order and duration (this alignment is inferred, not given), and true-label sentence rows from the Matlab files (12 subjects, 3,323 usable sentence rows).
- **Targets:** sentence embeddings from a MiniLM sentence-transformer, L2-normalised.
- **Decoder (identical for every representation):** standardisation fitted on training rows, ridge regression into the embedding space (strength chosen on validation subjects), cosine ranking over the unique test sentences.
- **Representations:** band-power features (whole sentence and 1.5 s chunks), the authors' band features, and a small contrastive 1D-CNN trained without labels on training subjects' training-sentence chunks only (8 epochs; two augmented views per chunk; noise, amplitude scaling, small shifts). An identical untrained encoder is the control.
- **Controls:** random input through the same pipeline, and a reading-length/duration-only baseline (no EEG).
- **Splits:** leave-subjects-out folds with 45 held-out test sentences and 45 validation sentences, so test subjects and test sentences are never in training; a leakage check runs on every fold. Identical sentences share one group.
- **Noise and quality:** white noise at 0, 0.5, 1 and 2 times each channel's standard deviation added to **test** EEG only; quality weighting down-weights training rows with a high artifact score.
- **Transfer:** a decoder trained on one dataset only, evaluated on the other (no target labels).
- **Metrics:** top-1, recall@K, MRR, median rank, `sim_margin`, with chance levels.

## Results
- **Sentence retrieval is at chance for every EEG representation.** In the main true-label experiment (45 candidates, 4 folds) EEG methods reach top-1 of 1.8-3.3% against 2.2% chance and 2.2% for random input; `sim_margin` is 0.0005-0.0086 (SD over folds 0.003-0.009) against 0.0000 for random input. The reading-length control alone gets 0.0093.
- **Continuous-EEG windows, unseen subjects and sentences:** ZuCo 1.0 top-1 2.8% (random 2.5%, chance 2.6%); ZuCo 2.0 2.0% (random 1.8%, chance 2.3%). Without holding out sentences (about 260-300 candidates) EEG is also at chance. A small ZuCo 1.0 effect after per-subject standardisation (`sim_margin` 0.0138, SD 0.0088 over 4 folds) did not appear in ZuCo 2.0 (-0.0020) and the length-only control is of similar size in the easier setting.
- **Self-supervised learning did not help.** The contrastive loss fell from about 3.5 to 1.9 over 8 epochs, so the encoder learned structure in the EEG, but its sentence retrieval (`sim_margin` 0.0005) equals the untrained encoder (0.0000).
- **Noise:** performance does not change with noise because there is no sentence information to lose; the curves are flat at chance.
- **Quality weighting:** no reliable difference (band-power chunks 0.0086 with weights vs 0.0071 without; encoder -0.0003 vs 0.0005; spread across folds about 0.005-0.009).
- **Unseen subjects:** the same chance-level result holds for held-out subjects in every fold; the spread across folds is reported above.
- **Cross-dataset transfer:** at chance in both directions (ZuCo 1.0 -> 2.0 top-1 0.48% vs 0.34% random; ZuCo 2.0 -> 1.0 0.53% vs 0.42% random).
- **Word level (48,638 word rows):** word retrieval equals the random-input control (top-1 3.9% vs 3.7%; reading-length control 4.8%), and sentence retrieval from word-level EEG is at chance (MRR 0.094 vs 0.095 random vs 0.098 chance).
- **Positive control:** the EEG features predict log first-fixation duration within a subject (r = 0.146, SD 0.072 over 12 subjects, random 0.012) and across subjects after per-subject standardisation (r = 0.082, random -0.007), but not word length (r about 0.008). So the pipeline can detect real information; what is present in these features is reading-related (eye-movement/reading-time), not sentence content.

## Did pretrained representations help?
Not tested: no pretrained EEG encoder was run. I did not find and verify a public pretrained model that accepts this 104-channel ZuCo data, and forcing an incompatible model would not be defensible.

## Conclusion
Within the limits below, the answer is no: sentence-level and first-fixation band-power EEG features, a ridge decoder and a small self-supervised encoder do not retrieve the correct sentence beyond random input and a reading-length control, for unseen subjects and across datasets. The detectable EEG information is related to reading behaviour. This is consistent with the concern raised in the literature that EEG-to-text results often do not exceed noise-input baselines, and it shows why the noise control is needed.

## Limitations
- Only sentence-level and first-fixation band features were tested; time-resolved word-level or other representations could behave differently, and a negative result here does not show that no language information exists in EEG.
- 4 folds, no subject-level bootstrap confidence intervals and no significance tests; small differences are within fold-to-fold variation.
- Per-subject standardisation is transductive and reported separately.
- The continuous-EEG sentence alignment is inferred; ZuCo 1.0 windows covered 264 of 300 sentences (cause not investigated); ZuCo 2.0 uses 8 of 18 subjects.
- Noise is synthetic and the quality score is a simple artifact fraction.
- The self-supervised encoder is small, trained for 8 epochs, and untuned.
- Word-retrieval chance is not meaningful because frequent words dominate; use the random-input control.