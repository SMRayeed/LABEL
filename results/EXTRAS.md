# W3PHIAI additional analyses

## 1. Retrieval helps only when the text names the fact

Accuracy closed book and with the drug's passage (%), and the change (points, 95% bootstrap CI over questions).

**OLMo-2-1124-7B-Instruct**

| Relation | Group | n | Closed | Passage | Change [95% CI] |
|---|---|---|---|---|---|
| indication | named | 150 | 75.3 | 88.0 | +12.7 [+5.3, +20.0] |
| indication | family only | 150 | 48.0 | 45.3 | -2.7 [-10.7, +5.3] |
| indication | silent | 150 | 44.7 | 43.3 | -1.3 [-9.3, +6.7] |
| contraindication | named | 150 | 42.0 | 54.7 | +12.7 [+3.3, +22.0] |
| contraindication | family only | 106 | 30.2 | 30.2 | +0.0 [-8.5, +8.5] |
| contraindication | silent | 150 | 26.0 | 20.7 | -5.3 [-12.7, +1.3] |

**Qwen2.5-0.5B-Instruct**

| Relation | Group | n | Closed | Passage | Change [95% CI] |
|---|---|---|---|---|---|
| indication | named | 150 | 47.3 | 86.7 | +39.3 [+31.3, +47.3] |
| indication | family only | 150 | 34.0 | 44.7 | +10.7 [+2.0, +20.0] |
| indication | silent | 150 | 26.7 | 34.0 | +7.3 [-0.7, +15.3] |
| contraindication | named | 150 | 30.0 | 70.7 | +40.7 [+32.0, +49.3] |
| contraindication | family only | 106 | 23.6 | 27.4 | +3.8 [-4.7, +12.3] |
| contraindication | silent | 150 | 29.3 | 23.3 | -6.0 [-13.3, +0.7] |

**Qwen2.5-7B-Instruct**

| Relation | Group | n | Closed | Passage | Change [95% CI] |
|---|---|---|---|---|---|
| indication | named | 150 | 78.7 | 85.3 | +6.7 [+0.0, +13.3] |
| indication | family only | 150 | 50.0 | 38.0 | -12.0 [-18.7, -5.3] |
| indication | silent | 150 | 39.3 | 22.7 | -16.7 [-24.0, -10.0] |
| contraindication | named | 150 | 33.3 | 44.0 | +10.7 [+0.7, +20.0] |
| contraindication | family only | 106 | 24.5 | 17.0 | -7.5 [-17.9, +2.8] |
| contraindication | silent | 150 | 17.3 | 3.3 | -14.0 [-20.0, -8.0] |

**granite-3.1-8b-instruct**

| Relation | Group | n | Closed | Passage | Change [95% CI] |
|---|---|---|---|---|---|
| indication | named | 150 | 76.7 | 90.7 | +14.0 [+7.3, +20.7] |
| indication | family only | 150 | 54.0 | 44.0 | -10.0 [-17.3, -2.7] |
| indication | silent | 150 | 45.3 | 29.3 | -16.0 [-24.0, -8.0] |
| contraindication | named | 150 | 18.7 | 38.0 | +19.3 [+10.7, +28.0] |
| contraindication | family only | 106 | 9.4 | 17.9 | +8.5 [+0.0, +17.0] |
| contraindication | silent | 150 | 9.3 | 2.0 | -7.3 [-12.0, -2.7] |

**Pooled over Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct, granite-3.1-8b-instruct** (each question averaged over models)

| Relation | Group | n | Change [95% CI] |
|---|---|---|---|
| indication | named | 150 | +11.1 [+6.7, +15.8] |
| indication | family only | 150 | -8.2 [-13.3, -3.1] |
| indication | silent | 150 | -11.3 [-17.1, -6.0] |
| contraindication | named | 150 | +14.2 [+7.3, +20.9] |
| contraindication | family only | 106 | +0.3 [-6.3, +6.9] |
| contraindication | silent | 150 | -8.9 [-12.9, -5.1] |

Named group, indication minus contraindication change: -3.1 points [-11.6, +5.1]

## 2. Scale curve (closed book, five options, chance 20%)

| Model | Indication | Off-label use | Contraindication | Ind. minus contra. [95% CI over drugs] |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 31.5 | 28.5 | 22.2 | +9.3 [+4.0, +14.9] |
| Qwen2.5-1.5B-Instruct | 45.7 | 35.8 | 25.3 | +20.3 [+15.2, +26.2] |
| Qwen2.5-3B-Instruct | 45.3 | 32.7 | 24.2 | +21.2 [+16.1, +26.9] |
| Qwen2.5-7B-Instruct | 50.8 | 35.0 | 23.8 | +27.0 [+21.6, +32.4] |
| Qwen2.5-14B-Instruct | 49.8 | 35.8 | 27.2 | +22.7 [+17.1, +28.3] |

## 3. FDA labels

Drugs with a label: 472 of 574.

| Relation | n | Wikipedia | Wiki+PubMed | Label, own section | Whole label | All three | Prose misses rescued by label section |
|---|---|---|---|---|---|---|---|
| contraindication | 1416 | 10.9 | 17.2 | 2.6 | 22.9 | 26.7 | 1.4% |
| indication | 1416 | 22.2 | 30.5 | 19.1 | 23.7 | 34.0 | 3.4% |

