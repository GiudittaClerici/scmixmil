# personal functions
import pandas as pd
import scanpy as sc
import anndata as ad
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Binomial, Categorical, LowRankMultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
from torch_scatter import scatter_softmax, segment_add_csr
from tqdm.auto import trange
from mixmil.anndata import prepare_anndata as prepare_mixmil_anndata


# Apply to each cell type
#adata_obs = adata_obs.groupby('cell_type', group_keys=False).apply(mark_top_bottom)

def mark_top_bottom(df, attention_col='attention_weight', group_col = 'cell_type', qmargin=0.05):
    def label_group(group):
        q_low = group[attention_col].quantile(qmargin)
        q_high = group[attention_col].quantile(1-qmargin)
        group[f'bottom{qmargin:.0%}'] = group[attention_col] <= q_low
        group[f'top{qmargin:.0%}'] = group[attention_col] >= q_high
        return group

    return df.groupby(group_col, group_keys=False).apply(label_group)

#df = mark_top_bottom(df, attention_col='attn_score')



@torch.inference_mode()
def get_raw_logits(mixmil_model, Xs, Fs, scaling=None):
        """
        Calculates the raw logits for the given input data.
        This combines the predicted bag-level latent effects (u) with fixed effects (Fs).
        """
        u = mixmil_model.predict(Xs, scaling=scaling) # shape: (N, P)
        # alpha shape: (K, P)
        # Fs shape: (N, K)
        fixed_effects_term = Fs.mm(mixmil_model.alpha) # shape: (N, P)

        # Logits need to be (N, P, 1) for binomial/binary categorical,
        # or (N, P, num_categories) for multi-categorical
        # Since u is (N, P) and fixed_effects_term is (N, P), their sum is (N, P).
        # We need to unsqueeze to match the expected (N, P, 1) or (N, P, num_categories) format for Categorical.
        logits = fixed_effects_term + u # This is (N, P)

        # Handle the binary categorical case where P=1 and logits needs to be (N, P, 2)
        if mixmil_model.likelihood_name == "categorical" and logits.shape[1] == 1:
             # This assumes a binary classification where P=1, and the single logit
             # represents the positive class, with the negative class being -logit.
             # You might need to adjust this logic if your P represents something else
             # for multi-class categorical, or if P > 1 and each P is a separate binary task.
            logits_reshaped = torch.cat([-logits, logits], dim=-1).unsqueeze(1) # (N, 1, 2) -> (N, P, 2)
        elif mixmil_model.likelihood_name == "categorical":
            # For multi-class categorical where P > 1 (e.g., P is directly the number of categories)
            # and each value in u/alpha corresponds to a category logit
            logits_reshaped = logits.unsqueeze(1) # (N, P) -> (N, 1, P) for single output categorical with P classes
        else: # Binomial
             logits_reshaped = logits.unsqueeze(-1) # (N, P) -> (N, P, 1) for binomial/binary where P is # of outputs

        return logits_reshaped

def get_predicted_labels(mixmil_model, Xs, Fs, scaling=None):
        """
        Returns the final predicted labels based on the likelihood.
        For categorical likelihood, this will be the class index with the highest probability.
        For binomial likelihood, this will be the predicted counts.
        """
        logits = get_raw_logits(mixmil_model, Xs, Fs, scaling=scaling)

        if mixmil_model.likelihood_name == "categorical":
            # The structure of `logits` from `get_raw_logits` is crucial here.
            # Based on the `likelihood` method, if `P=1`, then `logits` becomes `(N, 1, 2)`.
            # If `P > 1` (and it's a single categorical output with P classes), then `logits` becomes `(N, 1, P)`.
            # We want to apply softmax over the last dimension (the categories).

            if logits.shape[1] == 1 and logits.shape[2] == 2: # Binary categorical output (P=1)
                logits_for_softmax = logits.squeeze(1) # (N, 2)
            elif logits.shape[1] == 1 and logits.shape[2] > 2: # Multi-class categorical output (P=num_categories)
                logits_for_softmax = logits.squeeze(1) # (N, P)
            elif logits.shape[2] == 1 and logits.shape[1] > 1: # P independent binary categorical tasks
                # Logits shape (N, P_tasks, 1). Need to convert to (N, P_tasks, 2) for softmax
                logits_for_softmax = torch.cat([-logits, logits], dim=-1).view(-1, 2) # (N * P_tasks, 2)
                # This will return a flat array of labels, need to reshape later if P_tasks labels are desired.
            else:
                # Fallback for unexpected shapes, try to make it (N, num_categories)
                # This might need more specific logic depending on how P and the output categories interact.
                # For robustness, we assume the last dimension is always the categories for argmax.
                logits_for_softmax = logits.view(-1, logits.shape[-1])

            # Calculate probabilities using softmax over the categories dimension
            probabilities = torch.softmax(logits_for_softmax, dim=-1)

            # Get the index of the highest probability (the predicted label)
            predicted_labels = torch.argmax(probabilities, dim=-1)

            # If it was P independent binary tasks, reshape back to (N, P_tasks)
            if logits.shape[2] == 1 and logits.shape[1] > 1:
                predicted_labels = predicted_labels.view(logits.shape[0], logits.shape[1])

            return predicted_labels

        elif mixmil_model.likelihood_name == "binomial":
            # For binomial, logits are directly proportional to the probability of success
            # We convert logits to probabilities using sigmoid
            probs = torch.sigmoid(logits)
            # Predicted count is n_trials * probability
            predicted_counts = torch.round(probs * mixmil_model.n_trials)
            return predicted_counts.squeeze(-1) # Remove the last dim of 1

        else:
            raise ValueError(f"Unsupported likelihood: {mixmil_model.likelihood_name}")

            
def prepare_anndata(anndata, layer, FE_column_to_scale_IDXs, sex_col, FE_col, donor_column, target_column, scale_y ):
    fixed_effects = [] if FE_col is None else [x for x in FE_col if x not in {donor_column, 'null'}]
    scale_fixed_effects = None
    if FE_column_to_scale_IDXs != 'None':
        scale_fixed_effects = [fixed_effects[i] for i in FE_column_to_scale_IDXs if i < len(fixed_effects)]

    prepared = prepare_mixmil_anndata(
        anndata,
        bag_key=donor_column,
        target_key=target_column,
        feature_key=layer,
        feature_source='obsm',
        fixed_effects=fixed_effects,
        split_key='split',
        scale_fixed_effects=scale_fixed_effects,
        scale_target=scale_y,
    )
    if prepared.test is None:
        raise ValueError('The AnnData object does not contain any test bags')
    return (
        prepared.n_features,
        prepared.n_fixed_effects,
        prepared.train.Xs,
        prepared.train.F,
        prepared.train.Y,
        prepared.test.Xs,
        prepared.test.F,
        prepared.test.Y,
        *(prepared.target_scaling if prepared.target_scaling is not None else (None, None)),
        prepared.train.bag_ids,
        prepared.test.bag_ids,
        prepared.train.cell_ids,
        prepared.test.cell_ids,
    )


# from Jan Engelmann:
def to_device(el, device):
    """
    Move a nested structure of elements (dict, list, tuple, torch.Tensor, torch.nn.Module) to the specified device.

    Parameters:
    - el: Element or nested structure of elements to be moved to the device.
    - device (torch.device): The target device, such as 'cuda' for GPU or 'cpu' for CPU.

    Returns:
    - Transferred element(s) in the same structure: Elements moved to the specified device.
    """
    if isinstance(el, dict):
        return {k: to_device(v, device) for k, v in el.items()}
    elif isinstance(el, (list, tuple)):
        return [to_device(x, device) for x in el]
    elif isinstance(el, (torch.Tensor, torch.nn.Module)):
        return el.to(device)
    else:
        return el
    

def add_attention_weights_to_anndata(mil_model, anndata, Xs_train, Xs_test, before_softmax = 0, ):
    #before_softmax = 0 (False) -> it gets the weight at position 0, which is the one after softmax
    #before_softmax = 1 (True) -> it gets the one in position 1, which is the one before softmax 
    if before_softmax == 0:
        string = 'att_w_after_sfmx'
        print('Adding the attention weights after softmax')
    else: 
        string = 'att_w_before_sfmx'
        print('Adding the attention weights before softmax')
    train_adata = anndata[anndata.obs["split"] == "train"].copy()
    test_adata = anndata[anndata.obs["split"] == "test"].copy()
    train_w = mil_model.get_weights(Xs_train)[before_softmax]
    test_w = mil_model.get_weights(Xs_test)[before_softmax]

    def add_weights(subset, weights):
        values = torch.cat(weights, dim=0) if isinstance(weights, list) else weights
        values = values.detach().cpu().numpy()
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != subset.n_obs:
            raise ValueError('Attention weights are not aligned with AnnData observations')
        columns = [string] if values.shape[1] == 1 else [f'{string}_{i}' for i in range(values.shape[1])]
        subset.obs[columns] = pd.DataFrame(values, index=subset.obs_names, columns=columns)
        return subset

    return ad.concat([add_weights(train_adata, train_w), add_weights(test_adata, test_w)], merge="same")

# adata_both = add_attention_weights_to_anndata(mil_model = model,
#                                  anndata = adata_both, Xs_train = Xs, Xs_test = test_Xs, 
#                                  before_softmax = 0, )

def add_top_percent_flags(anndata, top_percent=5):
    """
    Adds new columns to anndata.obs for each attention weight column starting with 'att_w_'.
    Each new column marks whether the value is in the top `top_percent` percent.

    Parameters:
    - adata: AnnData object
    - top_percent: float, the percentage threshold (e.g., 5 for top 5%)

    Returns:
    - Modified AnnData with new boolean columns in .obs
    """
    top_color = "#800080"  # Purple for top X%
    other_color = "#FFD700"  # Yellow for others
    
    for col in anndata.obs.columns:
        if col.startswith('att_w_'):
            threshold = anndata.obs[col].quantile(1 - top_percent / 100)
            new_col = f'top{top_percent}%_for_weight_{col.replace("att_w_", "")}'
            anndata.obs[new_col] = anndata.obs[col] > threshold
            anndata.uns[f"{new_col}_colors"] = [other_color, top_color]
    return anndata

#adata = add_top_percent_flags(adata, top_percent=5)



def plot_umaps_topX_layers(adata, topX_cols, ncols=4,
                                 top_color="#800080", other_color="#D3D3D3", alpha_other=0.3):
    n_plots = len(topX_cols)
    nrows = -(-n_plots // ncols)  # ceiling division

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axs = axs.flatten()

    for i, col in enumerate(topX_cols):
        ax = axs[i]

        top = adata[adata.obs[col] == True].copy()
        other = adata[adata.obs[col] == False].copy()

        # Add dummy color obs for plotting
        top.obs["__tmp__"] = "top"
        other.obs["__tmp__"] = "other"
        combined = other.concatenate(top)

        # Define color map manually
        combined.uns["__tmp___colors"] = [other_color, top_color]

        sc.pl.umap(combined, color="__tmp__", ax=ax, show=False, title=col, size=25)
        ax.set_facecolor("white")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
    
def bin_continuous_column(df, column_name, bin_edges, bin_labels=None):
    """
    Bins a continuous column in a Pandas DataFrame according to specified bin edges.
    The output column name will be the original column name + '_binned'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the continuous column to bin.
        bin_edges (list-like): A sequence of bin edges, including the left edge of the first bin and the right edge of the last bin.
        bin_labels (list-like, optional): Labels to use for the resulting bins.
                                          Should be one fewer than the number of bin edges.
                                          If None, default interval labels will be used.

    Returns:
        pd.DataFrame: The DataFrame with an additional column containing the binned data.
    """
    output_column_name = f'{column_name}_binned'
    if bin_labels is None:
        bin_labels = [f'{bin_edges[i]}-{bin_edges[i+1]-1}' for i in range(len(bin_edges) - 1)]

    df[output_column_name] = pd.cut(df[column_name], bins=bin_edges, right=False, labels=bin_labels, include_lowest=True)

    return df




def plot_umaps_topX_with_alpha(adata, topX_cols, ncols=4,
                                top_color="#800080", 
                               other_color="#D3D3D3", alpha_other=0.1, string4output = 'test'):
    import matplotlib.pyplot as plt
    import numpy as np

    umap_coords = adata.obsm['X_umap']
    n_plots = len(topX_cols)
    nrows = -(-n_plots // ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axs = axs.flatten()

    for i, col in enumerate(topX_cols):
        ax = axs[i]

        top_mask = adata.obs[col] == True
        other_mask = ~top_mask

        # Plot "other" cells first with transparency
        ax.scatter(umap_coords[other_mask, 0], umap_coords[other_mask, 1],
                   c=other_color, alpha=alpha_other, s=10, label='Other')

        # Plot "top X%" cells on top
        ax.scatter(umap_coords[top_mask, 0], umap_coords[top_mask, 1],
                   c=top_color, alpha=1.0, s=10, label='Top X%')

        ax.set_title(col)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(string4output + 'top_attended_cells_umaps.png')
    plt.show()

# topX_cols = [col for col in adata_both.obs.columns if col.startswith("topX%_for_weight_")]
# plot_umaps_topX_layers_fixed(adata_both, topX_cols, ncols=4)
# plot_umaps_topX_with_alpha(adata_both, topX_cols, ncols=4)

def get_feature_importance(anndata, mil_model, layer=None):
    """Return the posterior means for the latent value-effect coefficients."""
    feature_importance = mil_model.qz_mu
    if isinstance(feature_importance, torch.Tensor):
        feature_importance = feature_importance.detach().cpu().numpy()

    return feature_importance

# feat_imp = get_feature_importance(anndata = adata_both, mil_model=model)

def plot_feature_importance(anndata, mil_model, string_for_output, n_cols):
    
    feat_imp = get_feature_importance(anndata, mil_model)
    n_features, n_classes = feat_imp.shape
    feature_names = anndata.var_names

    # Plot grid
    n_cols = n_cols
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

    for i in range(n_classes): # for each class, plot a barplot 
        ax = axes[i // n_cols, i % n_cols]
        ax.bar(range(n_features), feat_imp[:, i])
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
        ax.set_title(f"Class {i} Feature Importance")
        ax.set_ylabel("Importance")
        ax.set_xlabel("Features")

    # Hide unused axes
    for j in range(n_classes, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(string_for_output + 'multi_class_feature_importance.png')
    plt.show()
    

def calculate_metrics(y_true, y_pred_proba, y_pred_class): # Assuming predictions are probabilities for AUC
    # You might need to convert probabilities to class labels for some metrics
    # y_pred_class = (y_pred_proba > 0.5).astype(int) # Example threshold
    
    # function used to compute metrics in the run_classification_mixmil 
    metrics = {}
    metrics['AUC_ROC'] = roc_auc_score(y_true, y_pred_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    metrics['AUC_PR'] = auc(recall, precision) 
    metrics['Precision'] = precision_score(y_true, y_pred_class)
    metrics['Recall'] = recall_score(y_true, y_pred_class)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['F1_macro'] = f1_score(y_true, y_pred_class, average='macro')

    # For TP, FP, TN, FN
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred_class).ravel()
    metrics['TN'] = TN
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TP'] = TP
    metrics['Support'] = len(y_true) # Or sum of TP+FN, or specific support metric

    return metrics
