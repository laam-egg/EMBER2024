DOWNLOAD_MODELS_ONLY = False
import sys
if len(sys.argv) >= 2 and sys.argv[1] == "--models-only":
    DOWNLOAD_MODELS_ONLY = True

import os
import thrember

MODELS_DIR = os.path.abspath("../models")
DATASET_DIR = os.path.abspath("../dataset")

try:
    os.mkdir(MODELS_DIR)
except FileExistsError:
    pass

if not DOWNLOAD_MODELS_ONLY:
    try:
        os.mkdir(DATASET_DIR)
    except FileExistsError:
        pass

thrember.download_models(MODELS_DIR)

if not DOWNLOAD_MODELS_ONLY:
    thrember.download_dataset(DATASET_DIR)
