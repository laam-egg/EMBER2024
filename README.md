# EMBER2024 Instructions

## Setup

```sh
cd $PROJECT_ROOT
pip install .
pip install -r myscripts/requirements.txt
```

## Utility Scripts

```sh
cd my_scripts
python download.py
python load_datasets.py
python vectorize_win32.py
```

Actually just download the models,
then run inference:

```sh
cd my_scripts
python inference.py
```

**To extract features for use in pefe-system,**
first create a new file named `config.json`
under `my_scripts` that follows the format
of `config.example.json`. Then:

```sh
cd my_scripts
python extract_features.py
```

To run inference on those extracted features:

```sh
cd my_scripts
python infer_extracted_features.py
```

## Evaluate LGBM

```sh
cd examples
python eval_lgbm.py "../dataset" "../models/EMBER2024_all.model"
open Classifier_ROC_AUC.pdf
```
