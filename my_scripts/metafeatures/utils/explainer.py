"""
xAI
"""

from typing import Any, Literal, override
import numpy as np
import shap

class AbstractModel:
    def predict_sample(self, feature_vector: np.ndarray) -> float:
        raise NotImplementedError
    
    def get_raw_model_instance(self) -> Any:
        raise NotImplementedError

from dataclasses import dataclass, asdict
from typing import Type, TypeVar

T = TypeVar("T", bound="AbstractRecord")

@dataclass
class AbstractRecord:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        return cls(**d)

class AbstractModelExplainer:
    def __init__(
        self,
        model: AbstractModel,
    ):
        self.model = model

    def compute_shap_for_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError










def logit(y):
    # Ensure y is within the valid range (0, 1) to avoid math errors
    if not (0 < np.min(y) and np.max(y) < 1):
        raise ValueError("Input 'y' must be in the range (0, 1)")
        
    return np.log(y / (1 - y))

def sigmoid(x):
    """
    Computes the element-wise sigmoid of x.

    x: A single number, a NumPy array, a vector, or a matrix.
    Returns: The sigmoid value or array of values between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))

def logit_to_odds_multiplier(delta_logit: float) -> float:
    return np.exp(delta_logit)

def format_odds_change(delta_logit: float) -> str:
    mult = np.exp(delta_logit)

    if mult >= 1:
        return f"↑ INCREASES malware odds by {mult:.1f}×"
    else:
        return f"↓ REDUCES malware odds by {1/mult:.1f}×"



class TreeModelExplainer(AbstractModelExplainer):
    def __init__(
        self,
        model: AbstractModel,
    ):
        super().__init__(model)
        self.explainer = shap.TreeExplainer(
            self.model.get_raw_model_instance()
        )
    
    @override
    def compute_shap_for_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        shap_values = self.explainer.shap_values(feature_matrix)
        return np.array(shap_values)
