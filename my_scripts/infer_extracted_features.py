import lightgbm as lgb
import thrember
from pathlib import Path
import os
import time
import lmdb
from pefe_agent.config import *
import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

THRESHOLD = 0.5

FRAME_SIZE = 2048

MODELS_PATH = Path("../models")

def main():
    # Step 0: Load the LightGBM model
    model = lgb.Booster(model_file=MODELS_PATH / "EMBER2024_all.model")

    # Step 1: Load the data
    db = lmdb.open(config['self']['lmdb_path'], readonly=True, lock=False, map_size=1024 * 1024 * 1024 * 1024) # 1 TB

    with db.begin() as txn:
        cursor = txn.cursor()
        vectors_in_frame = []
        labels_in_frame = []
        frame_number = 1

        TOTAL_HITS = 0
        TOTAL_MISSES = 0
        TOTAL_COUNT = 0

        TOTAL_TP = 0
        TOTAL_TN = 0
        TOTAL_FP = 0
        TOTAL_FN = 0

        TOTAL_TRUE = 0
        TOTAL_FALSE = 0

        def process_frame(vectors_in_frame, labels_in_frame):
            nonlocal frame_number, TOTAL_HITS, TOTAL_MISSES, TOTAL_COUNT
            nonlocal TOTAL_TP, TOTAL_TN, TOTAL_FP, TOTAL_FN
            nonlocal TOTAL_TRUE, TOTAL_FALSE

            try:
                frame_input = np.stack(vectors_in_frame, axis=0)
                frame_labels = np.array(labels_in_frame)
                frame_output = (model.predict(frame_input) >= THRESHOLD).astype(int)

                hits = np.sum(frame_output == frame_labels)
                misses = np.sum(frame_output != frame_labels)
                tp = np.sum((frame_output == 1) & (frame_labels == 1))
                fp = np.sum((frame_output == 1) & (frame_labels == 0))
                fn = np.sum((frame_output == 0) & (frame_labels == 1))
                tn = np.sum((frame_output == 0) & (frame_labels == 0))

                frame_size = len(vectors_in_frame)
                assert hits + misses == frame_size

                num_true = np.sum((frame_labels == 1))
                num_false = np.sum((frame_labels == 0))
                assert num_true + num_false == frame_size

                print(f"Frame #{frame_number} (size={frame_size} = {num_true} positive + {num_false} negative samples):\n    hits={hits}, misses={misses}, tp={tp}, fp={fp}, fn={fn}, tn={tn}")

                vectors_in_frame.clear()
                labels_in_frame.clear()

                TOTAL_HITS += hits
                TOTAL_MISSES += misses
                TOTAL_COUNT += frame_size
                TOTAL_TP += tp
                TOTAL_TN += tn
                TOTAL_FP += fp
                TOTAL_FN += fn
                TOTAL_TRUE += num_true
                TOTAL_FALSE += num_false
            finally:
                frame_number += 1

        c = 0
        for k, v in cursor:
            c += 1
            payload = msgpack.unpackb(v, raw=False)
            label = payload['lb']       # int
            features = payload['ef']    # numpy array

            labels_in_frame.append(label)
            vectors_in_frame.append(features)

            if len(vectors_in_frame) >= FRAME_SIZE:
                process_frame(vectors_in_frame, labels_in_frame)
        
        process_frame(vectors_in_frame, labels_in_frame)

    print("DONE.")

    accuracy = TOTAL_HITS / TOTAL_COUNT if TOTAL_COUNT > 0 else 0.0
    precision = TOTAL_TP / (TOTAL_TP + TOTAL_FP) if (TOTAL_TP + TOTAL_FP) > 0 else 0.0
    recall = TOTAL_TP / (TOTAL_TP + TOTAL_FN) if (TOTAL_TP + TOTAL_FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("============== DATASET ================")
    print(f"TOTAL: {TOTAL_COUNT}")
    print(f"    = {TOTAL_TRUE} positive samples")
    print(f"    + {TOTAL_FALSE} negative samples")
    print()
    print("============= INFERENCE ===============")
    print(f"TOTAL HITS: {TOTAL_HITS}")
    print(f"TOTAL MISSES: {TOTAL_MISSES}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"TP: {TOTAL_TP}, TN: {TOTAL_TN}, FP: {TOTAL_FP}, FN: {TOTAL_FN}")
    print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1-score: {f1:.2%}")

    assert TOTAL_COUNT == c
    assert TOTAL_HITS + TOTAL_MISSES == TOTAL_COUNT
    print("No anomalies found.")

if __name__ == "__main__":
    main()
