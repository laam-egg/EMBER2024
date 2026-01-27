# EMBER2024 Instructions

- [EMBER2024 Instructions](#ember2024-instructions)
  - [Setup](#setup)
  - [Utility Scripts](#utility-scripts)
    - [Download the Models](#download-the-models)
    - [Vectorize the Dataset](#vectorize-the-dataset)
    - [Quick Inference](#quick-inference)
    - [Mass Feature Extraction](#mass-feature-extraction)
    - [Mass Inference and Evaluation](#mass-inference-and-evaluation)
  - [Packing Inference Model into EXE](#packing-inference-model-into-exe)
  - [xAI](#xai)
  - [Appendix: Evaluate LGBM](#appendix-evaluate-lgbm)

## Setup

```sh
cd $PROJECT_ROOT
pip install .
pip install -r myscripts/requirements.txt
```

## Utility Scripts

### Download the Models

This is the prerequisite
for all the next steps.

```sh
cd my_scripts
../venv/bin/activate
python download.py --models-only
```

If you wish to download
EMBER2024 (vectorized) dataset
also, run this command
instead:

```sh
python download.py
```

But if you just want to
run inference and evaluate
on your own custom dataset,
you don't need that.

### Vectorize the Dataset

The original authors provide
[a way to do that](./README-original.md#vectorizing-raw-features).
But oh god it requires 44.8 GB of RAM ?

So I made a script that vectorize
the dataset into - well not a numpy
data file - but a SQLite3 database file.
[Check it out.](./my_scripts/vectorize.py).

Or, download the vectorized dataset
[here](https://www.kaggle.com/datasets/laamegg/ember2024-dataset-sqlite3)

### Quick Inference

If you have a couple of files in a directory
and want to test them out real quick:

```sh
python inference.py /dir/containing/files/to/infer
```

But if you have thousands of files,
maybe you should go through standardized
steps: Extract Features, then Infer and
Evaluate. Follow the next sections.

### Mass Feature Extraction

Which is, extract features from
many, e.g. thousands of files
at once.

**To extract features for use in pefe-system,**
first create a new file named `config.json`
under `my_scripts` that follows the format
of `config.example.json`. Then:

```sh
cd my_scripts
../venv/bin/activate
python extract_features.py
```

### Mass Inference and Evaluation

Which is, infer from above extracted features,
i.e. from many, e.g. thousands of files
at once, then evaluate the results.

1. Using old scripts
    
    - Quick
    - Results are printed directly to console
    - No visualization

    ```sh
    cd my_scripts
    ../venv/bin/activate
    python infer_extracted_features.py
    # OR:
    python infer_and_evaluate.py
    ```

2. Using new scripts
    
    - More thorough, polished evaluation with visualizations
    - Need a separate tool in `pefe-system` to view the results
        (results are not printed directly to console).
    
    First, run this for inference and evaluation on
    extracted features ("IEF"):

    ```sh
    cd my_scripts
    ../venv/bin/activate
    python -m new_ief
    ```

    **It is assumed that the model files**
    **are located in** `$PROJECT_ROOT/models`.
    If they are not, you could create a symlink
    to the actual location, instead of copying
    them which is time-consuming.

    Then, to view those results, use
    [the tool `pefe-ief-viz`](https://github.com/pefe-system/pefe-ief-viz).
    The `RESULTS_DIR` is already set
    correctly for you. If not, set it
    to `$PROJECT_ROOT/my_scripts/RESULTS`
    (the path is also printed to console when
    the command finishes).

    If you don't want to run it yourself,
    the visualization notebook and HTML file,
    which I ran against my own dataset,
    are also available in `$PROJECT_ROOT/my_scripts/visualization`.
    To run that notebook or export it again to HTML, though, you still
    need to follow `pefe-ief-viz`'s instructions.

    **It is a known issue that the HTML file**
    **might not display properly when hosted**
    **and accessed online,** you might have to
    download it instead.

## Packing Inference Model into EXE

On Windows, install PyInstaller and UPX.

Then, create a new Python virtual environment,
and install minimal dependencies (for reduced
EXE file size):

```powershell
pip install -r requirements-inference-exe.txt
```

Then run PyInstaller (**from project root**) as
follows - with your actual UPX installation
directory filled in:

```powershell
pyinstaller `
  --exclude-module matplotlib `
  --exclude-module cycler `
  --exclude-module fonttools `
  --exclude-module contourpy `
  --exclude-module kiwisolver `
  --onefile .\my_scripts\inference_exe.py `
  --add-data "models;models" `
  --add-data "venv\Lib\site-packages\mscerts;mscerts" `
  --upx-dir <path\to\UPX\installation\directory>
```

The resulting EXE file is about 75 MB.
Run it with flag `--help` for usage.

## xAI

[See this file](./my_scripts/xAI/README.md).

## Appendix: Evaluate LGBM

The script by original authors.

```sh
cd examples
python eval_lgbm.py "../dataset" "../models/EMBER2024_all.model"
open Classifier_ROC_AUC.pdf
```
