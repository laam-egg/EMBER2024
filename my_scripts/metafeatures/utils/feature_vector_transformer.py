import numpy as np
from typing import List, Dict, Tuple, Set, Callable


class FeatureMap:
    def __init__(self) -> None:
        self.ordered_feature_names: List[str] = []
        self.feature_name_to_index: Dict[str, int] = {}
        self.immutable = False
    
    def set_immutable(self, immutable: bool) -> None:
        self.immutable = immutable

    def set_feature_names(self, feature_names: List[str]) -> None:
        if self.immutable:
            raise RuntimeError("Cannot modify an immutable FeatureMap.")
        # Defensive copy
        self.ordered_feature_names = list(feature_names)
        self.feature_name_to_index = {
            name: idx for idx, name in enumerate(self.ordered_feature_names)
        }

    def dim(self) -> int:
        return len(self.ordered_feature_names)
    
    def get_feature_name_by_index(self, index: int) -> str:
        return self.ordered_feature_names[index]
    
    def get_feature_index_by_name(self, name: str) -> int:
        return self.feature_name_to_index[name]


class FeatureVectorView:
    """
    Immutable read-only view over a feature vector.
    Always refers to the ORIGINAL feature map.
    """
    def __init__(self, feature_vector: np.ndarray, feature_map: FeatureMap) -> None:
        self._vector = feature_vector
        self._map = feature_map

    def get(self, feature_name: str):
        idx = self._map.feature_name_to_index.get(feature_name)
        if idx is None:
            raise KeyError(f"Feature '{feature_name}' not found.")
        return float(self._vector[idx])


# ---------------- Alterations ---------------- #

class FeatureVectorAlteration:
    pass


class FeatureAddition(FeatureVectorAlteration):
    def __init__(self, feature_name: str | None = None) -> None:
        if feature_name is None:
            feature_name = self.__class__.__name__
        self.feature_name = feature_name

    def apply(self, vector_view: FeatureVectorView) -> float:
        """
        Must be implemented by subclasses.
        Should ONLY read from vector_view.
        """
        raise NotImplementedError


class FeatureRemoval(FeatureVectorAlteration):
    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name


# ---------------- Transformer ---------------- #

class CompiledFeatureTransform:
    """
    Fully compiled, validation-free transformer.
    Safe to apply repeatedly to many vectors.
    """
    def __init__(
        self,
        kept_indices: np.ndarray,
        add_fns: List[Callable[[np.ndarray], float]],
        feature_map: FeatureMap,
    ) -> None:
        self.kept_indices = kept_indices
        self.add_fns = add_fns
        self.feature_map = feature_map
        feature_map.set_immutable(True)

        self._k = len(kept_indices)
        self._m = len(add_fns)
        self._out_dim = feature_map.dim()
    
    def get_feature_map(self) -> FeatureMap:
        return self.feature_map

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        out = np.empty(self._out_dim, dtype=vector.dtype)

        # Fast slice
        out[:self._k] = vector[self.kept_indices]

        # Compute added features
        for i, fn in enumerate(self.add_fns):
            out[self._k + i] = fn(vector)

        return out

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (N, D)
        """
        N = X.shape[0]
        out = np.empty((N, self._out_dim), dtype=X.dtype)

        out[:, :self._k] = X[:, self.kept_indices]

        for i, fn in enumerate(self.add_fns):
            col = self._k + i
            for r in range(N):
                out[r, col] = fn(X[r])

        return out

class FeatureVectorTransformer:
    def __init__(
        self,
        original_feature_names: List[str],
        alterations: List[FeatureVectorAlteration],
    ) -> None:
        self.alterations = list(alterations)

        self.original_feature_names = list(original_feature_names)
        self.original_map = FeatureMap()
        self.original_map.set_feature_names(self.original_feature_names)
        self.original_map.set_immutable(True)

    def compile(self) -> CompiledFeatureTransform:
        removed = set()
        added = []
        added_names = set()
        add_fns = []

        # Validation + intention collection
        for alt in self.alterations:
            if isinstance(alt, FeatureRemoval):
                name = alt.feature_name

                if name in added_names:
                    raise ValueError(f"Cannot remove added feature '{name}'.")
                if name not in self.original_map.feature_name_to_index:
                    raise KeyError(f"Unknown feature '{name}'.")
                if name in removed:
                    raise ValueError(f"Feature '{name}' removed multiple times.")

                removed.add(name)

            elif isinstance(alt, FeatureAddition):
                name = alt.feature_name

                if name in added_names:
                    raise ValueError(f"Feature '{name}' added multiple times.")
                if name in self.original_map.feature_name_to_index:
                    raise ValueError(f"Feature '{name}' already exists.")
                if name in removed:
                    raise ValueError(f"Feature '{name}' was removed earlier.")

                # Capture a *pure* function
                def make_fn(addition: FeatureAddition):
                    def fn(vec: np.ndarray):
                        view = FeatureVectorView(vec, self.original_map)
                        return addition.apply(view)
                    return fn

                add_fns.append(make_fn(alt))
                added.append(name)
                added_names.add(name)

            else:
                raise TypeError(f"Unknown alteration: {type(alt).__name__}")

        # Resolve kept indices (NO STRINGS AT RUNTIME)
        kept_indices = np.array(
            [
                self.original_map.feature_name_to_index[name]
                for name in self.original_feature_names
                if name not in removed
            ],
            dtype=np.int64,
        )

        # Build final feature map
        final_names = [
            name for name in self.original_feature_names
            if name not in removed
        ] + added

        feature_map = FeatureMap()
        feature_map.set_feature_names(final_names)
        feature_map.set_immutable(True)

        return CompiledFeatureTransform(
            kept_indices=kept_indices,
            add_fns=add_fns,
            feature_map=feature_map,
        )
