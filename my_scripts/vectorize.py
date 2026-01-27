"""
After downloading the dataset, you receive JSONL files under a directory.

Given that directory path i.e. the dataset path, this script processes all JSONL files
of a specified type (train, test, challenge) and extracts feature vectors from each entry.

It then stores these feature vectors along with their corresponding SHA256 hashes and labels
into a SQLite3 database for efficient retrieval and analysis.
"""

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vectorize_win32.py <dataset_path> <type> <sqlite3_path>")
        print("  <dataset_path>: Path to the dataset directory")
        print("  <type>: Type of dataset (e.g., 'train', 'test', 'challenge')")
        print("  <sqlite3_path>: Path to save the SQLite3 database")
        sys.exit(1)

import os
import thrember
import json
import sqlite3

extractor = thrember.PEFeatureExtractor()

def extract_features(raw_bytes: bytes):
    # Step 1: Extract features (raw features in dict form)
    features = extractor.raw_features(raw_bytes)

    # Step 2: Vectorize the features into a numeric array
    X = extractor.process_raw_features(features)  # vectorize() expects a list
    
    # Step 3: Return
    return X.tobytes()

def extract_all(input_dir: str, type: str, db_path: str):
    # Connect to SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                sha256 TEXT PRIMARY KEY,
                label INTEGER,
                feature_vector BLOB
            )
        """)
        conn.commit()

        if type == "challenge":
            SUFFIX = f"challenge_malicious.jsonl"
        else:
            SUFFIX = f"{type}.jsonl"

        for root, _, files in os.walk(input_dir):
            for file_name in files:
                if not file_name.endswith(SUFFIX):
                    continue
                
                jsonl_file_path = os.path.join(root, file_name)
                print(f"Processing {jsonl_file_path}...")

                with open(jsonl_file_path, 'r') as f:
                    for line in f:
                        raw_obj = json.loads(line)
                        X = extractor.process_raw_features(raw_obj)
                        feature_vector = X.tobytes()
                        sha256 = raw_obj.get('sha256', None)
                        label = raw_obj.get('label', None)
                        if sha256 is None:
                            print("Skipping entry without sha256")
                            continue
                        if label is None:
                            print(f"Skipping entry {sha256} without label")
                            continue

                        try:
                            # Store hash and key (hex) in database
                            cursor.execute(
                                "INSERT INTO files (sha256, label, feature_vector) VALUES (?, ?, ?)",
                                (sha256, label, feature_vector)
                            )
                            print(".", end="", flush=True)
                        except Exception as e:
                            print(f"Skipping entry {sha256} due to error: {e}")
        
                conn.commit()
                print(f"\nFinished processing {jsonl_file_path}")
        
        print(f"All files processed. Database saved to {db_path}")
        print(f"Creating index for the `label` column... ", end="", flush=True)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_label ON files(label);"
        )
        print(f"Done.")

if __name__ == "__main__":
    DATASET_PATH = os.path.abspath(sys.argv[1])
    TYPE = sys.argv[2]
    SQLITE3_PATH = os.path.abspath(sys.argv[3])

    print(f"DATASET_PATH = {DATASET_PATH}")
    print(f"TYPE = {TYPE}")
    print(f"SQLITE3_PATH = {SQLITE3_PATH}")
    print("=================================")

    extract_all(DATASET_PATH, TYPE, SQLITE3_PATH)
