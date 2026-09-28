import numpy as np
import pytest
import torch

ad = pytest.importorskip("anndata")

from mixmil.anndata import add_attention_to_anndata, prepare_anndata


@pytest.fixture
def small_adata():
    obs = {
        "donor": ["d1", "d1", "d2", "d2", "d3"],
        "split": ["train", "train", "train", "train", "test"],
        "target": [0, 0, 1, 1, 0],
        "target2": [1, 1, 0, 0, 1],
        "age": [20, 20, 40, 40, 30],
        "sex": [0, 0, 1, 1, 0],
    }
    result = ad.AnnData(
        X=np.zeros((5, 2), dtype=np.float32),
        obs=obs,
    )
    result.obsm["embedding"] = np.arange(15, dtype=np.float32).reshape(5, 3)
    return result


def test_prepare_anndata_preserves_bags_and_cells(small_adata):
    prepared = prepare_anndata(
        small_adata,
        bag_key="donor",
        target_key="target",
        feature_key="embedding",
        fixed_effects=["sex", "age"],
        split_key="split",
        scale_fixed_effects=["age"],
    )

    assert prepared.train.bag_ids == ["d1", "d2"]
    assert prepared.test.bag_ids == ["d3"]
    assert prepared.train.cell_ids == [["0", "1"], ["2", "3"]]
    assert prepared.train.Xs[0].shape == (2, 3)
    assert prepared.train.F.shape == (2, 3)  # intercept, sex, age
    assert prepared.test.Y.tolist() == [[0.0]]


def test_attention_is_aligned_by_cell_id(small_adata):
    prepared = prepare_anndata(
        small_adata,
        bag_key="donor",
        target_key="target",
        feature_key="embedding",
        split_key="split",
    )
    attention = [torch.tensor([[0.1], [0.9]]), torch.tensor([[0.2], [0.8]])]
    result = add_attention_to_anndata(small_adata, prepared.train, attention)

    assert result.obs.loc["0", "mixmil_attention"] == pytest.approx(0.1)
    assert result.obs.loc["1", "mixmil_attention"] == pytest.approx(0.9)
    assert np.isnan(result.obs.loc["4", "mixmil_attention"])


def test_prepare_anndata_supports_multiple_targets(small_adata):
    prepared = prepare_anndata(
        small_adata,
        bag_key="donor",
        target_key=["target", "target2"],
        feature_key="embedding",
        split_key="split",
    )
    assert prepared.train.Y.shape == (2, 2)
    assert prepared.target_name == ("target", "target2")
