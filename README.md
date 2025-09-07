# EMBER2024 Instructions

- [EMBER2024 Instructions](#ember2024-instructions)
  - [Setup](#setup)
  - [Utility Scripts](#utility-scripts)
    - [Download the Models](#download-the-models)
    - [Quick Inference](#quick-inference)
    - [Mass Feature Extraction](#mass-feature-extraction)
    - [Mass Inference and Evaluation](#mass-inference-and-evaluation)
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

## Appendix: Evaluate LGBM

The script by original authors.

```sh
cd examples
python eval_lgbm.py "../dataset" "../models/EMBER2024_all.model"
open Classifier_ROC_AUC.pdf
```
