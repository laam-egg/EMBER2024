# EMBER2024 Instructions

## Utility Scripts

```sh
python download.py
python load_datasets.py
python vectorize_win32.py
```

## Evaluate LGBM

```sh
cd examples
python eval_lgbm.py "../dataset" "../models/EMBER2024_all.model"
open Classifier_ROC_AUC.pdf
```
