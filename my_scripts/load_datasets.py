import os
import thrember

DATASET_PATH = os.path.abspath("../dataset")

print("Loading X_train, y_train...")
X_train, y_train = thrember.read_vectorized_features(DATASET_PATH, subset="train")

print("Loading X_test, y_test...")
X_test, y_test = thrember.read_vectorized_features(DATASET_PATH, subset="test")

# print("Loading X_challenge, y_challenge...")
# X_challenge, y_challenge = thrember.read_vectorized_features(DATASET_PATH, subset="challenge")
