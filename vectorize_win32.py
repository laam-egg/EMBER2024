import os
import thrember

DATASET_PATH = os.path.abspath("../dataset")

thrember.create_vectorized_features(DATASET_PATH)
