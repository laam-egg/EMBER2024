from utils.feature_vector_transformer import FeatureVectorTransformer, FeatureVectorAlteration, FeatureAddition
from utils.original_features import ORIGINAL_FEATURES_LIST
from typing import List
import numpy as np
from .model_trainer import ModelTrainer
from .explainer import TreeModelExplainer, AbstractModel
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score

@dataclass
class NewFeatureImpact:
    name: str
    global_shap_abs: float

    """Ranking 4 means the 4th greatest contribution."""
    global_shap_abs_ranking: int
    
    """75% percentile means the feature has more contribution than 75% of the rest.
    If a feature is top X% performer, then X = 100 - this value"""
    global_shap_abs_percentile: float

    global_shap_abs_delta_from_average: float
    global_shap_abs_delta_from_max: float
    global_shap_abs_delta_from_min: float

@dataclass
class MetafeatureExperimentResult:
    auc: float
    new_feature_impacts: List[NewFeatureImpact]

    def __str__(self) -> str:
        lines = [f"AUC: {self.auc:.12f}"]
        lines.append("New Feature Impacts:")
        for impact in self.new_feature_impacts:
            lines.append(
                f"  - {impact.name}: E(|SHAP|)={impact.global_shap_abs:.6f}, "
                f"Ranking={impact.global_shap_abs_ranking}, "
                f"Percentile={impact.global_shap_abs_percentile:.2f}%, "
                f"Delta from Avg={impact.global_shap_abs_delta_from_average:.6f}, "
                f"Delta from Max={impact.global_shap_abs_delta_from_max:.6f}, "
                f"Delta from Min={impact.global_shap_abs_delta_from_min:.6f}"
            )
        else:
            if not self.new_feature_impacts:
                lines.append("  (no new features)")
        return "\n".join(lines)

class MetafeatureExperiment:
    def __init__(
        self,
        alterations: List[FeatureVectorAlteration],
    ) -> None:
        self.transformer = FeatureVectorTransformer(
            ORIGINAL_FEATURES_LIST,
            alterations=alterations,
        ).compile()

        self.new_feature_names: list[str] = list(
            filter(
                lambda x: x != "",
                map(
                    lambda alt: alt.feature_name if isinstance(alt, FeatureAddition) else "",
                    alterations,
                ),
            )
        )
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> MetafeatureExperimentResult:
        def log(msg: str) -> None:
            print(f"[MetafeatureExperiment] {msg}", flush=True)
        
        log("Transforming training set...")
        X_train = self.transformer.transform_batch(X_train)
        log("Transforming validation set...")
        X_val = self.transformer.transform_batch(X_val)

        log("Training model...")
        model = ModelTrainer({
            "objective": "binary",
            "metric": "auc",

            "early_stopping_rounds": 10,

            "n_estimators": 100,
            "num_leaves": 128,
            "max_depth": -1, # let num_leaves control complexity

            "learning_rate": 0.1,
            "min_data_in_leaf": 50,

            "verbosity": -1,
        }).train(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        del X_train
        del X_val

        log("Transforming test set...")
        X_test = self.transformer.transform_batch(X_test)

        log("Running model predictions on test set...")
        test_preds: np.ndarray = model.predict(X_test) # type: ignore
        assert len(test_preds.shape) == 1 and test_preds.shape[0] == X_test.shape[0]

        class JustTrainedModel(AbstractModel):
            def predict_sample(self, feature_vector: np.ndarray) -> float:
                a2d = np.array([feature_vector])
                pred = model.predict(a2d)[0][1] # type: ignore
                return pred
            
            def get_raw_model_instance(self):
                nonlocal model
                return model

        log("Computing SHAP values on test set...")
        explainer = TreeModelExplainer(
            model=JustTrainedModel(),
        )
        all_shap_vals = explainer.compute_shap_for_batch(X_test)

        log("Computing global SHAP values (for test set)...")
        global_shap_abses = np.abs(all_shap_vals).mean(axis=0)

        log("Computing AUC on test set...")
        auc = float(roc_auc_score(y_test, test_preds))

        log("Computing new feature impacts...")
        new_feature_impacts = []
        new_feature_shap_abses = []
        feature_map = self.transformer.get_feature_map()
        for feature_name in self.new_feature_names:
            feature_index = feature_map.get_feature_index_by_name(feature_name)
            feature_shap_abses = global_shap_abses[feature_index]
            new_feature_shap_abses.append((feature_name, feature_shap_abses))
        # Compute statistics
        max_shap_abs = max(global_shap_abses)
        min_shap_abs = min(global_shap_abses)
        avg_shap_abs = sum(global_shap_abses) / len(global_shap_abses)
        N = len(global_shap_abses)
        for feature_name, feature_shap_val in new_feature_shap_abses:
            ranking = 1 + np.sum(global_shap_abses > feature_shap_val)
            percentile = 100.0 * np.sum(global_shap_abses < feature_shap_val) / N

            impact = NewFeatureImpact(
                name=feature_name,
                global_shap_abs=feature_shap_val,
                global_shap_abs_ranking=ranking,
                global_shap_abs_percentile=percentile,
                global_shap_abs_delta_from_average=feature_shap_val - avg_shap_abs,
                global_shap_abs_delta_from_max=feature_shap_val - max_shap_abs,
                global_shap_abs_delta_from_min=feature_shap_val - min_shap_abs,
            )
            new_feature_impacts.append(impact)
        
        log("Experiment completed.")
        return MetafeatureExperimentResult(
            auc=auc,
            new_feature_impacts=new_feature_impacts,
        )
