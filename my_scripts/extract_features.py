import thrember
from pefe_agent.consumer import PEFEConsumer
import hashlib

def inspect(filename):
    with open(filename, "rb") as f:
        raw_bytes = f.read()

    id = hashlib.sha256(raw_bytes).digest()

    # Step 1: Extract features (raw features in dict form)
    extractor = thrember.PEFeatureExtractor()
    features = extractor.raw_features(raw_bytes)

    # Step 2: Vectorize the features into a numeric array
    X = extractor.process_raw_features(features)  # vectorize() expects a list
    
    # Step 3: Return
    return id, X

class EMBER2024PEFEConsumer(PEFEConsumer):
    def handle_pe_file(self, path):
        return inspect(path)

def main():
    EMBER2024PEFEConsumer("EMBER2024PEFE").run()

if __name__ == "__main__":
    main()
