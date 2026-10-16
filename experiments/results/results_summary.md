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
