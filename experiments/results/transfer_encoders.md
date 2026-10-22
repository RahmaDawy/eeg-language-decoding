**Cross-dataset transfer, source-only decoder; SSL rows are means over the 4 source checkpoints (mrr_sd = SD across them)**

| source | target | target_subset | representation | method | n_target_queries | n_candidates | top1 | chance_top1 | mrr | chance_mrr | mrr_sd | sim_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zuco1 | zuco2 | unseen_sentences | bandpower | plain | 1726 | 255.0000 | 0.0035 | 0.0039 | 0.0244 | 0.0240 | nan | 0.0010 |
| zuco1 | zuco2 | all_sentences | bandpower | plain | 1993 | 293.0000 | 0.0040 | 0.0034 | 0.0221 | 0.0214 | nan | 0.0013 |
| zuco1 | zuco2 | unseen_sentences | cbramod | frozen_pretrained | 1726 | 255.0000 | 0.0017 | 0.0039 | 0.0215 | 0.0240 | nan | -0.0022 |
| zuco1 | zuco2 | all_sentences | cbramod | frozen_pretrained | 1993 | 293.0000 | 0.0035 | 0.0034 | 0.0206 | 0.0214 | nan | -0.0020 |
| zuco1 | zuco2 | unseen_sentences | cbramod | untrained_cbramod | 1726 | 255.0000 | 0.0023 | 0.0039 | 0.0219 | 0.0240 | nan | -0.0025 |
| zuco1 | zuco2 | all_sentences | cbramod | untrained_cbramod | 1993 | 293.0000 | 0.0045 | 0.0034 | 0.0218 | 0.0214 | nan | -0.0023 |
| zuco1 | zuco2 | unseen_sentences | ssl | ssl_trained | 1726 | 255.0000 | 0.0039 | 0.0039 | 0.0229 | 0.0240 | 0.0002 | -0.0022 |
| zuco1 | zuco2 | all_sentences | ssl | ssl_trained | 1993 | 293.0000 | 0.0035 | 0.0034 | 0.0210 | 0.0214 | 0.0001 | -0.0023 |
| zuco1 | zuco2 | unseen_sentences | ssl | untrained_encoder | 1726 | 255.0000 | 0.0039 | 0.0039 | 0.0226 | 0.0240 | 0.0010 | -0.0022 |
| zuco1 | zuco2 | all_sentences | ssl | untrained_encoder | 1993 | 293.0000 | 0.0034 | 0.0034 | 0.0208 | 0.0214 | 0.0004 | -0.0018 |
| zuco1 | zuco2 | unseen_sentences | random | random_input | 1726 | 255.0000 | 0.0042 | 0.0039 | 0.0230 | 0.0240 | 0.0002 | -0.0031 |
| zuco1 | zuco2 | all_sentences | random | random_input | 1993 | 293.0000 | 0.0037 | 0.0034 | 0.0208 | 0.0214 | 0.0001 | -0.0033 |
| zuco2 | zuco1 | unseen_sentences | bandpower | plain | 2548 | 226.0000 | 0.0063 | 0.0044 | 0.0293 | 0.0265 | nan | 0.0032 |
| zuco2 | zuco1 | all_sentences | bandpower | plain | 2992 | 264.0000 | 0.0037 | 0.0038 | 0.0248 | 0.0233 | nan | 0.0026 |
| zuco2 | zuco1 | unseen_sentences | cbramod | frozen_pretrained | 2548 | 226.0000 | 0.0012 | 0.0044 | 0.0232 | 0.0265 | nan | -0.0013 |
| zuco2 | zuco1 | all_sentences | cbramod | frozen_pretrained | 2992 | 264.0000 | 0.0037 | 0.0038 | 0.0219 | 0.0233 | nan | -0.0011 |
| zuco2 | zuco1 | unseen_sentences | cbramod | untrained_cbramod | 2548 | 226.0000 | 0.0027 | 0.0044 | 0.0237 | 0.0265 | nan | -0.0007 |
| zuco2 | zuco1 | all_sentences | cbramod | untrained_cbramod | 2992 | 264.0000 | 0.0033 | 0.0038 | 0.0216 | 0.0233 | nan | -0.0008 |
| zuco2 | zuco1 | unseen_sentences | ssl | ssl_trained | 2548 | 226.0000 | 0.0032 | 0.0044 | 0.0251 | 0.0265 | 0.0021 | -0.0007 |
| zuco2 | zuco1 | all_sentences | ssl | ssl_trained | 2992 | 264.0000 | 0.0033 | 0.0038 | 0.0224 | 0.0233 | 0.0009 | -0.0003 |
| zuco2 | zuco1 | unseen_sentences | ssl | untrained_encoder | 2548 | 226.0000 | 0.0024 | 0.0044 | 0.0240 | 0.0265 | 0.0010 | -0.0012 |
| zuco2 | zuco1 | all_sentences | ssl | untrained_encoder | 2992 | 264.0000 | 0.0033 | 0.0038 | 0.0220 | 0.0233 | 0.0002 | -0.0008 |
| zuco2 | zuco1 | unseen_sentences | random | random_input | 2548 | 226.0000 | 0.0037 | 0.0044 | 0.0258 | 0.0265 | 0.0018 | 0.0002 |
| zuco2 | zuco1 | all_sentences | random | random_input | 2992 | 264.0000 | 0.0039 | 0.0038 | 0.0237 | 0.0233 | 0.0015 | 0.0005 |
