"""AnnData integration for MixMIL.

The model itself operates on PyTorch tensors.  This module provides the
conversion layer needed for scverse workflows while keeping cell and bag
identifiers attached to the converted data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def _to_numpy(value: Any) -> np.ndarray:
    """Convert an AnnData/pandas/sparse value to a dense NumPy array."""
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def _as_numeric(values: Any, name: str) -> np.ndarray:
    values = _to_numpy(values)
    if np.issubdtype(values.dtype, np.number):
        result = values.astype(np.float32, copy=False)
    else:
        try:
            result = values.astype(np.float32)
        except (TypeError, ValueError):
            categories, result = np.unique(values.astype(str), return_inverse=True)
            if len(categories) == 0:
                raise ValueError(f"{name!r} does not contain any values")
            result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name!r} contains missing or non-finite values")
    return result


def _resolve_features(adata: Any, feature_source: str, feature_key: Optional[str]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if feature_source == "X":
        matrix = adata.X
        names = tuple(str(x) for x in getattr(adata, "var_names", range(matrix.shape[1])))
    elif feature_source == "layers":
        if feature_key is None:
            raise ValueError("feature_key is required when feature_source='layers'")
        matrix = adata.layers[feature_key]
        names = tuple(str(x) for x in getattr(adata, "var_names", range(matrix.shape[1])))
    elif feature_source == "obsm":
        if feature_key is None:
            raise ValueError("feature_key is required when feature_source='obsm'")
        matrix = adata.obsm[feature_key]
        columns = getattr(matrix, "columns", None)
        names = tuple(str(x) for x in columns) if columns is not None else ()
    else:
        raise ValueError("feature_source must be one of 'X', 'layers', or 'obsm'")

    matrix = _to_numpy(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"MixMIL features must be two-dimensional, got shape {matrix.shape}")
    if matrix.shape[0] != adata.n_obs:
        raise ValueError("The feature matrix must have one row per AnnData observation")
    try:
        matrix = matrix.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("MixMIL features must be numeric") from exc
    if not np.isfinite(matrix).all():
        raise ValueError("Features contain missing or non-finite values")
    if not names:
        names = tuple(f"feature_{i}" for i in range(matrix.shape[1]))
    return matrix, names


def _first_unique(values: Sequence[Any]) -> Any:
    first = values[0]
    for value in values[1:]:
        if str(value) != str(first):
            raise ValueError("Values expected to be constant within each bag are inconsistent")
    return first


@dataclass
class BaggedData:
    """Tensor data for a set of bags, with identifiers retained."""

    Xs: List[torch.Tensor]
    F: torch.Tensor
    Y: torch.Tensor
    bag_ids: List[str]
    cell_ids: List[List[str]]

    def __len__(self) -> int:
        return len(self.bag_ids)

    @property
    def n_features(self) -> int:
        return int(self.Xs[0].shape[1]) if self.Xs else 0

    def to(self, device: Any) -> "BaggedData":
        self.Xs = [x.to(device) for x in self.Xs]
        self.F = self.F.to(device)
        self.Y = self.Y.to(device)
        return self


@dataclass
class AnnDataMixMILData:
    """Train/test bagged tensors produced from an AnnData object."""

    train: BaggedData
    test: Optional[BaggedData]
    feature_names: Tuple[str, ...]
    fixed_effect_names: Tuple[str, ...]
    target_name: Union[str, Tuple[str, ...]]
    bag_key: str
    feature_source: str
    feature_key: Optional[str]
    scaling_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    target_scaling: Optional[Tuple[float, float]] = None

    @property
    def n_features(self) -> int:
        return self.train.n_features

    @property
    def n_fixed_effects(self) -> int:
        return int(self.train.F.shape[1])


def _encode_fixed_effects(adata: Any, bag_ids: List[str], bag_rows: Dict[str, np.ndarray], columns: Sequence[str]):
    import pandas as pd

    if not columns:
        return np.ones((len(bag_ids), 1), dtype=np.float32), ["intercept"]

    values = {}
    for column in columns:
        if column not in adata.obs.columns:
            raise KeyError(f"Fixed-effect column {column!r} was not found in adata.obs")
        series = adata.obs[column]
        values[column] = [_first_unique(series.iloc[bag_rows[bag]].tolist()) for bag in bag_ids]

    frame = pd.DataFrame(values, index=bag_ids)
    encoded = []
    names = ["intercept"]
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            array = series.to_numpy(dtype=np.float32)[:, None]
            if not np.isfinite(array).all():
                raise ValueError(f"Fixed-effect column {column!r} contains missing or non-finite values")
            encoded.append(array)
            names.append(column)
        else:
            dummy = pd.get_dummies(series.astype(str), prefix=column, dtype=np.float32)
            encoded.append(dummy.to_numpy(dtype=np.float32))
            names.extend(str(x) for x in dummy.columns)

    result = np.concatenate([np.ones((len(frame), 1), dtype=np.float32)] + encoded, axis=1)
    return result, names


def _make_split(
    features: np.ndarray,
    obs_names: np.ndarray,
    bag_values: np.ndarray,
    target_values: np.ndarray,
    fixed_values: np.ndarray,
    bag_order: Sequence[str],
    mask: np.ndarray,
    dtype: torch.dtype,
) -> BaggedData:
    selected_bags = [bag for bag in bag_order if np.any(mask & (bag_values == bag))]
    Xs, Ys, cell_ids = [], [], []
    for bag in selected_bags:
        rows = np.flatnonzero(mask & (bag_values == bag))
        Xs.append(torch.as_tensor(features[rows], dtype=dtype))
        Ys.append(target_values[bag_order.index(bag), :])
        cell_ids.append([str(x) for x in obs_names[rows]])
    F = torch.as_tensor(fixed_values[[bag_order.index(bag) for bag in selected_bags]], dtype=dtype)
    Y = torch.as_tensor(np.asarray(Ys, dtype=np.float32), dtype=dtype)
    return BaggedData(Xs=Xs, F=F, Y=Y, bag_ids=selected_bags, cell_ids=cell_ids)


def prepare_anndata(
    adata: Any,
    bag_key: str,
    target_key: Union[str, Sequence[str]],
    feature_key: Optional[str] = None,
    feature_source: str = "obsm",
    fixed_effects: Optional[Sequence[str]] = None,
    split_key: Optional[str] = None,
    train_value: Any = "train",
    test_value: Any = "test",
    scale_fixed_effects: Optional[Sequence[str]] = None,
    scale_target: bool = False,
    dtype: torch.dtype = torch.float32,
) -> AnnDataMixMILData:
    """Convert an AnnData object into donor/bag-level MixMIL inputs.

    The input object is never modified.  Outcomes and fixed effects must be
    constant within each bag.  Categorical fixed effects are one-hot encoded;
    numeric fixed effects are retained as numeric columns.
    """
    if bag_key not in adata.obs.columns:
        raise KeyError(f"Bag column {bag_key!r} was not found in adata.obs")
    target_keys = [target_key] if isinstance(target_key, str) else list(target_key)
    if not target_keys:
        raise ValueError("At least one target column is required")
    missing_targets = [key for key in target_keys if key not in adata.obs.columns]
    if missing_targets:
        raise KeyError(f"Target columns were not found in adata.obs: {missing_targets}")
    if not getattr(adata.obs_names, "is_unique", len(set(adata.obs_names)) == adata.n_obs):
        raise ValueError("AnnData observation names must be unique")

    features, feature_names = _resolve_features(adata, feature_source, feature_key)
    obs_names = np.asarray([str(x) for x in adata.obs_names])
    bag_values = np.asarray([str(x) for x in adata.obs[bag_key].to_numpy()])
    target_values = np.column_stack([_as_numeric(adata.obs[key].to_numpy(), key) for key in target_keys])

    bag_order = list(dict.fromkeys(bag_values.tolist()))
    bag_rows = {bag: np.flatnonzero(bag_values == bag) for bag in bag_order}
    bag_targets = np.asarray([target_values[rows[0], :] for rows in bag_rows.values()], dtype=np.float32)
    for bag_index, (bag, rows) in enumerate(bag_rows.items()):
        if not np.allclose(target_values[rows, :], bag_targets[bag_index, :]):
            raise ValueError(f"Target values are inconsistent within bag {bag!r}")

    fixed_effects = list(fixed_effects or [])
    fixed_values, fixed_names = _encode_fixed_effects(adata, bag_order, bag_rows, fixed_effects)
    scaling_params = {}
    for column in scale_fixed_effects or []:
        if column not in fixed_names:
            raise KeyError(f"Scaled fixed effect {column!r} is not an encoded fixed-effect column")
        index = fixed_names.index(column)
        train_mask_for_scaling = np.ones(len(bag_order), dtype=bool)
        if split_key is not None:
            if split_key not in adata.obs.columns:
                raise KeyError(f"Split column {split_key!r} was not found in adata.obs")
            split_by_bag = np.asarray([
                str(_first_unique(adata.obs.iloc[bag_rows[bag]][split_key].tolist())) for bag in bag_order
            ])
            train_mask_for_scaling = split_by_bag == str(train_value)
        mean = float(fixed_values[train_mask_for_scaling, index].mean())
        std = float(fixed_values[train_mask_for_scaling, index].std())
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"Cannot scale fixed effect {column!r}")
        std = max(std, 1e-8)
        fixed_values[:, index] = (fixed_values[:, index] - mean) / std
        scaling_params[column] = (mean, std)

    if split_key is None:
        train_mask = np.ones(adata.n_obs, dtype=bool)
        test_mask = np.zeros(adata.n_obs, dtype=bool)
    else:
        if split_key not in adata.obs.columns:
            raise KeyError(f"Split column {split_key!r} was not found in adata.obs")
        split_values = np.asarray([str(x) for x in adata.obs[split_key].to_numpy()])
        train_mask = split_values == str(train_value)
        test_mask = split_values == str(test_value)
        if not train_mask.any():
            raise ValueError(f"No observations found for train split {train_value!r}")

        train_bags = set(bag_values[train_mask])
        test_bags = set(bag_values[test_mask])
        overlap = train_bags.intersection(test_bags)
        if overlap:
            raise ValueError(f"Bags occur in both train and test splits: {sorted(overlap)[:5]}")

    train = _make_split(features, obs_names, bag_values, bag_targets, fixed_values, bag_order, train_mask, dtype)
    test = None
    if test_mask.any():
        test = _make_split(features, obs_names, bag_values, bag_targets, fixed_values, bag_order, test_mask, dtype)

    target_scaling = None
    if scale_target:
        mean = train.Y.mean(0)
        std = train.Y.std(0).clamp_min(1e-8)
        train.Y = (train.Y - mean) / std
        if test is not None:
            test.Y = (test.Y - mean) / std
        target_scaling = (mean.tolist(), std.tolist())

    return AnnDataMixMILData(
        train=train,
        test=test,
        feature_names=feature_names,
        fixed_effect_names=tuple(fixed_names),
        target_name=target_keys[0] if len(target_keys) == 1 else tuple(target_keys),
        bag_key=bag_key,
        feature_source=feature_source,
        feature_key=feature_key,
        scaling_params=scaling_params,
        target_scaling=target_scaling,
    )


def add_attention_to_anndata(
    adata: Any,
    bagged: BaggedData,
    attention: Any,
    key: str = "mixmil_attention",
    inplace: bool = False,
) -> Any:
    """Store attention weights in ``adata.obs`` aligned by observation name."""
    result = adata if inplace else adata.copy()
    if isinstance(attention, torch.Tensor):
        if attention.ndim == 2:
            attention = attention[:, :, None]
        attention = [attention[i] for i in range(attention.shape[0])]
    arrays = [_to_numpy(value) for value in attention]
    if len(arrays) != len(bagged.cell_ids):
        raise ValueError("The number of attention arrays must match the number of bags")
    n_outputs = arrays[0].shape[1] if arrays and arrays[0].ndim == 2 else 1
    cell_to_row = {str(name): i for i, name in enumerate(result.obs_names)}
    for output in range(n_outputs):
        values = np.full(result.n_obs, np.nan, dtype=np.float32)
        for cells, array in zip(bagged.cell_ids, arrays):
            if array.shape[0] != len(cells):
                raise ValueError("Attention weights are not aligned with bag cell IDs")
            current = array[:, output] if n_outputs > 1 else array.reshape(-1)
            for cell, value in zip(cells, current):
                if cell not in cell_to_row:
                    raise KeyError(f"Cell {cell!r} was not found in the target AnnData object")
                values[cell_to_row[cell]] = value
        result.obs[key if n_outputs == 1 else f"{key}_{output}"] = values
    return result


def add_bag_predictions_to_anndata(
    adata: Any,
    bag_ids: Sequence[str],
    predictions: Any,
    key: str = "mixmil",
    inplace: bool = False,
) -> Any:
    """Store bag-level predictions in ``adata.uns`` with their bag IDs."""
    result = adata if inplace else adata.copy()
    values = _to_numpy(predictions)
    result.uns[key] = {
        "bag_ids": [str(x) for x in bag_ids],
        "predictions": values.tolist(),
    }
    return result
