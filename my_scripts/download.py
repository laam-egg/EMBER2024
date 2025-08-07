import os
import thrember

MODELS_DIR = os.path.abspath("../models")
DATASET_DIR = os.path.abspath("../dataset")

try:
    os.mkdir(MODELS_DIR)
except FileExistsError:
    pass

try:
    os.mkdir(DATASET_DIR)
except FileExistsError:
    pass

thrember.download_models(MODELS_DIR)

thrember.download_dataset(DATASET_DIR)
