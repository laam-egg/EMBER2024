import sys
root_dir = sys.argv[1]

import lightgbm as lgb
import thrember
from pathlib import Path
import time
import os

THRESHOLD = 0.5

MODELS_PATH = Path("../models")

# Step 0: Load the LightGBM model
model = lgb.Booster(model_file=MODELS_PATH / "EMBER2024_all.model")

def inspect(filename):
    with open(filename, "rb") as f:
        raw_bytes = f.read()

    start_time = time.perf_counter()
    # Step 1: Extract features (raw features in dict form)
    extractor = thrember.PEFeatureExtractor()
    features = extractor.raw_features(raw_bytes)

    # Step 2: Vectorize the features into a numeric array
    X = extractor.process_raw_features(features)  # vectorize() expects a list
    # print("Feature vector:", len(X), X)

    # Step 4: Predict
    score = model.predict([X])[0]

    end_time = time.perf_counter()

    duration = end_time - start_time
    
    return score, score >= THRESHOLD, duration, len(raw_bytes)

malware_count = 0
total_count = 0
total_inference_duration = 0
total_bytes = 0

for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
    for filename in filenames:
        # if filename.lower().endswith(".exe"):
            total_count += 1
            full_path = os.path.join(dirpath, filename)
            print(f"Inspecting {full_path} ... ", end="", flush=True)
            score, is_malware, duration, num_bytes = inspect(full_path)
            speed = num_bytes / duration
            total_inference_duration += duration
            total_bytes += num_bytes
            if is_malware:
                malware_count += 1
                print(f"MALWARE | {score} ", end="", flush=True)
            else:
                print(f"BENIGN  | {score} ", end="", flush=True)
            print(f"[{num_bytes} (B) / {duration:.3f} (s)] ", end="", flush=True)
            print()

print("Total count:", total_count)
print("Malware count:", malware_count)
print("Malware percent: %.2f%%" % (malware_count / total_count * 100))
print("Total inference duration: %.3f (s)" % total_inference_duration)
print("Total bytes of PE files inferred: %d" % total_bytes)
print(f"Average inference speed: {(total_bytes / total_inference_duration):.3f} (B/s)")
