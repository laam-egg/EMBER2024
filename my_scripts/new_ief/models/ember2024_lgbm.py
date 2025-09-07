from .ember2024_model import EMBER2024_Model
import lightgbm as lgb

class EMBER2024_LGBM(EMBER2024_Model):
    def do_load(self, model_path):
        self._model = lgb.Booster(model_file=model_path)
    
    def do_predict(self, feature_vectors):
        y_probs = self._model.predict(feature_vectors)
        return y_probs

    def do_get_batch_size(self):
        return 65536
