import lightgbm as lgb
import thrember
from pathlib import Path

MODELS_PATH = Path("../models")

# Step 0: Load the LightGBM model
model = lgb.Booster(model_file=MODELS_PATH / "EMBER2024_all.model")

files = [
    ["/home/lam/Desktop/Viettel/MORE_DATA/MalDICT/disarmed_behavior_train/VirusShare_c9a0b3afb01bb38f76cb6dba115932a0", 1],
    ["/home/lam/Desktop/Viettel/MORE_DATA/DikeDataset/files/benign/0a8deb24eef193e13c691190758c349776eab1cd65fba7b5dae77c7ee9fcc906.exe", 0],
]

# Load the raw PE file
for file in files:
    with open(file[0], "rb") as f:
        raw_bytes = f.read()

    # Step 1: Extract features (raw features in dict form)
    extractor = thrember.PEFeatureExtractor()
    features = extractor.raw_features(raw_bytes)

    # Step 2: Vectorize the features into a numeric array
    X = extractor.process_raw_features(features)  # vectorize() expects a list
    # print("Feature vector:", len(X), X)

    # Step 4: Predict
    score = model.predict([X])[0]

    print(
        "Malware probability:", score,
        "| Output:", "MALWARE" if score >= 0.5 else "BENIGN",
        "| Target:", "MALWARE" if file[1] == 1 else "BENIGN"
    )
