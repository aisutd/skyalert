# Flood detection evaluation: `flood-unet-20261119T025417Z-107bfbad`

Generated 2026-11-19T03:08:33+00:00 from code `e4c7091a48f108441b526f6bf91d25254412b83e-dirty`. Dataset GEOID-Flood v1.1, split version `453b236e65a7`.
Input: one post-event Sentinel-1 GRD image (VV, VH). Output: per-pixel background / permanent water / flooded water at the acquisition time. This is observed flood extent, not a forecast, and a historical evaluation does not establish live coverage.

Selected on val: **U-Net alone** (val flood IoU: model alone 0.260, with permanent-water reference 0.185). Radar threshold baseline tuned on val: VH < -22.0 dB (val flood IoU 0.138).

## test: held-out events from the main tree (event + touching-area groups never seen in training)

110 tiles, 47 events (44 with flood). Valid coverage 61.5%. Tiles were selected with flooded tiles over-represented (see manifest_meta.json), so pooled pixel metrics describe this sample, not a natural flood prevalence.

| method | flood IoU | Dice | precision | recall | mean event IoU | dry-scene alarms | permanent water called flood | flood called permanent water | flood km² true / predicted |
|---|---|---|---|---|---|---|---|---|---|
| **U-Net (U-Net alone)** | 0.276 | 0.433 | 31.3% | 70.5% | 0.268 | 31/44 | 52.9% | 8.0% | 257.2 / 579.54 |
| U-Net alone | 0.276 | 0.433 | 31.3% | 70.5% | 0.268 | 31/44 | 52.9% | 8.0% | 257.2 / 579.54 |
| U-Net + permanent-water reference | 0.372 | 0.542 | 43.4% | 72.3% | 0.369 | 28/44 | 1.9% | 6.2% | 257.2 / 429.02 |
| VH < -22.0 dB + reference | 0.264 | 0.418 | 32.2% | 59.7% | 0.251 | 44/44 | 0.8% | 5.9% | 257.2 / 476.71 |
| never flood | 0.000 | 0.000 | n/a | 0.0% | 0.000 | 0/44 | 0.0% | 0.0% | 257.2 / 0.0 |

Dry-scene alarm: at least 0.05 km² mapped as flood on a tile with no labelled flood.

![test examples](examples_test.png)

Per event (chosen variant):

| event | tiles | flood IoU | Dice | flood px true | flood px predicted |
|---|---|---|---|---|---|
| EMSR166 | 2 | dry | - | 0 | 1515 |
| EMSR167 | 4 | 0.004 | 0.007 | 24629 | 111659 |
| EMSR192 | 4 | 0.273 | 0.428 | 74890 | 94527 |
| EMSR199 | 3 | 0.107 | 0.193 | 69617 | 329501 |
| EMSR258 | 3 | 0.123 | 0.219 | 17546 | 83870 |
| EMSR260 | 3 | 0.149 | 0.259 | 45622 | 55719 |
| EMSR267 | 3 | 0.236 | 0.382 | 228983 | 848061 |
| EMSR273 | 3 | 0.508 | 0.674 | 162703 | 236533 |
| EMSR275 | 3 | 0.432 | 0.603 | 192091 | 310645 |
| EMSR292 | 3 | 0.163 | 0.281 | 41192 | 156568 |
| EMSR333 | 2 | 0.251 | 0.401 | 26943 | 84304 |
| EMSR352 | 3 | 0.181 | 0.306 | 158648 | 836025 |
| EMSR358 | 3 | 0.313 | 0.477 | 39087 | 121070 |
| EMSR359 | 3 | 0.026 | 0.050 | 26840 | 11185 |
| EMSR411 | 3 | 0.349 | 0.517 | 34443 | 96922 |
| EMSR442 | 3 | 0.032 | 0.062 | 19670 | 266229 |
| EMSR467 | 3 | 0.031 | 0.061 | 15449 | 52090 |
| EMSR468 | 3 | 0.051 | 0.097 | 80669 | 55202 |
| EMSR486 | 1 | dry | - | 0 | 19034 |
| EMSR487 | 1 | 0.560 | 0.718 | 101696 | 67779 |
| EMSR497 | 2 | 0.343 | 0.511 | 102827 | 296423 |
| EMSR498 | 3 | 0.356 | 0.525 | 32570 | 65694 |
| EMSR501 | 3 | 0.619 | 0.765 | 308480 | 326258 |
| EMSR507 | 3 | 0.047 | 0.090 | 48871 | 9276 |
| EMSR517 | 3 | 0.267 | 0.422 | 23319 | 47748 |
| EMSR518 | 3 | 0.343 | 0.510 | 36154 | 89898 |
| EMSR520 | 1 | 0.327 | 0.493 | 41114 | 116972 |
| EMSR572 | 2 | 0.180 | 0.305 | 8514 | 35138 |
| EMSR659 | 2 | 0.027 | 0.053 | 25921 | 1616 |
| EMSR663 | 1 | 0.510 | 0.676 | 217491 | 144757 |
| EMSR664 | 2 | 0.375 | 0.545 | 47982 | 55796 |
| EMSR665 | 2 | 0.058 | 0.109 | 29183 | 15501 |
| EMSR680 | 2 | 0.509 | 0.675 | 9631 | 16744 |
| EMSR722 | 1 | 0.328 | 0.494 | 13004 | 38679 |
| EMSR728 | 2 | 0.121 | 0.217 | 7701 | 59570 |
| EMSR755 | 2 | 0.074 | 0.138 | 6067 | 18941 |
| EMSR756 | 2 | 0.507 | 0.673 | 41047 | 71787 |
| EMSR762 | 2 | 0.523 | 0.687 | 61944 | 59277 |
| EMSR763 | 2 | 0.472 | 0.642 | 17114 | 35817 |
| EMSR766 | 2 | 0.078 | 0.144 | 2858 | 31722 |
| EMSR768 | 2 | 0.205 | 0.340 | 6288 | 16404 |
| EMSR770 | 1 | dry | - | 0 | 2259 |
| EMSR771 | 2 | 0.510 | 0.675 | 59415 | 101870 |
| EMSR797 | 2 | 0.280 | 0.438 | 9963 | 30264 |
| EMSR850 | 2 | 0.058 | 0.110 | 11915 | 192846 |
| EMSR854 | 2 | 0.339 | 0.507 | 7065 | 16356 |
| EMSR856 | 1 | 0.541 | 0.702 | 34881 | 59321 |

## heldout: GEOID-Flood held-out activations EMSR857-871 (Jan-Mar 2026), later than every training event; tiles overlapping main-tree ground removed

50 tiles, 7 events (7 with flood). Valid coverage 61.4%. Tiles were selected with flooded tiles over-represented (see manifest_meta.json), so pooled pixel metrics describe this sample, not a natural flood prevalence.

| method | flood IoU | Dice | precision | recall | mean event IoU | dry-scene alarms | permanent water called flood | flood called permanent water | flood km² true / predicted |
|---|---|---|---|---|---|---|---|---|---|
| **U-Net (U-Net alone)** | 0.423 | 0.594 | 51.9% | 69.5% | 0.421 | 12/20 | 35.3% | 8.5% | 123.05 / 164.75 |
| U-Net alone | 0.423 | 0.594 | 51.9% | 69.5% | 0.421 | 12/20 | 35.3% | 8.5% | 123.05 / 164.75 |
| U-Net + permanent-water reference | 0.501 | 0.667 | 62.1% | 72.2% | 0.474 | 9/20 | 0.0% | 5.8% | 123.05 / 143.0 |
| VH < -22.0 dB + reference | 0.402 | 0.574 | 53.0% | 62.5% | 0.369 | 18/20 | 0.0% | 4.5% | 123.05 / 145.07 |
| never flood | 0.000 | 0.000 | n/a | 0.0% | 0.000 | 0/20 | 0.0% | 0.0% | 123.05 / 0.0 |

Dry-scene alarm: at least 0.05 km² mapped as flood on a tile with no labelled flood.

![heldout examples](examples_heldout.png)

Per event (chosen variant):

| event | tiles | flood IoU | Dice | flood px true | flood px predicted |
|---|---|---|---|---|---|
| EMSR857 | 9 | 0.484 | 0.652 | 205962 | 367869 |
| EMSR861 | 9 | 0.416 | 0.587 | 270670 | 382080 |
| EMSR864 | 8 | 0.369 | 0.539 | 218333 | 176365 |
| EMSR865 | 4 | 0.387 | 0.558 | 142920 | 265876 |
| EMSR866 | 7 | 0.437 | 0.609 | 143487 | 190362 |
| EMSR870 | 6 | 0.436 | 0.607 | 155048 | 179254 |
| EMSR871 | 7 | 0.415 | 0.587 | 94126 | 85644 |

## Limitations

- Single post-event image: flood vs permanent water relies on what the model learned about river and lake context, or on the static permanent-water reference. No pre-event change detection.
- Labels are CEMS Rapid Mapping delineations; label is not masked under cloud, but radar sees through cloud.
- The source dataset is provisional pending the T2 dataset decision; a different source needs its own adapter and re-evaluation.
- Scores are argmax classes, not calibrated probabilities.
