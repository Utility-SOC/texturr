
Banking77 train split (labels used only for scoring), 10 trials x 8 intents x 30 queries (embedding model all-MiniLM-L6-v2). True number of groups = 8.

| Method | ARI (mean ± sd) | NMI (mean ± sd) |
|---|---|---|
| TF-IDF + KMeans (true k) | 0.44 ± 0.09 | 0.64 ± 0.07 |
| texturr, fixed k=5 (old default) | 0.56 ± 0.06 | 0.78 ± 0.05 |
| texturr, auto k | 0.84 ± 0.14 | 0.90 ± 0.08 |
| texturr, true k (upper bound) | 0.87 ± 0.12 | 0.91 ± 0.08 |

Auto k chose: [8, 4, 11, 8, 9, 8, 7, 8, 9, 8] (true 8).
