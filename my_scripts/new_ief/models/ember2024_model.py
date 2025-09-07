from pefe_ief.models.abstract_model import AbstractModel
import thrember


class EMBER2024_Model(AbstractModel):
    def __init__(self):
        super().__init__()
        self._extractor = thrember.PEFeatureExtractor()
    
    @property
    def extractor(self):
        return self._extractor

    def do_extract_features(self, bytes):
        raw_features = self.extractor.raw_features(bytes)
        feature_vector = self.extractor.process_raw_features(raw_features)
        return feature_vector
