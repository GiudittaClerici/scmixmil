import seaborn as sns
import pandas as pd
import numpy as np
import random
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
# torch and model related
from tqdm import tqdm
import torch
from torch import Tensor
from typing import Optional
from mixmil import MixMIL
from mixmil.data import to_device
from mixmil.anndata import (
    add_attention_to_anndata,
    add_bag_predictions_to_anndata,
    prepare_anndata as prepare_mixmil_anndata,
)
# sklearn and stat
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, accuracy_score, f1_score, auc, average_precision_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
import scipy.stats as st
# other 
import site
import time
import os 
import sys
import gc
import pickle

import argparse
import logging

def train_and_predict(X_train, X_test, y_train, y_test, description = ''):
    clf = LogisticRegressionCV(
                        cv=3,
                        max_iter=1000,
                        class_weight = 'balanced'
                    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    return y_pred, y_pred_proba
        

def topXodds_ratio_compute(
    preds: Tensor,
    target: Tensor,
    covariates: Tensor,
    weights: Optional[Tensor] = None,
    top_x: Optional[int] = 1,
    top: Optional[bool] = True,
    return_all: Optional[bool] = False,
    n_groups: Optional[int] = None,
    #     mode: DataType = None,
    num_classes: Optional[int] = None,
    #     pos_label: Optional[int] = None,
    average: Optional[str] = None,
    #     sample_weights: Optional[Sequence] = None,
    task: Optional[str] = "binary",
) -> Tensor:
    """Computes Top X % Odds ratio.
    Args:
        preds: predictions from model (logits or probabilities)
        target: Ground truth labels
        top_x: top x out of n_groups
        n_groups: number of groups / bins to split the predictions into
        top: whether to go from top or bottom

        unused:
        mode: 'multi class multi dim' or 'multi-label' or 'binary'
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class.
            Should be set to ``None`` for binary problems
        average: Defines the reduction that is applied to the output:
        sample_weights: sample weights for each data point
        num_classes: accepted only to have a valid signature with other classes and for checks
        
    Explained:
        Computes odds ratio for the top X% predicted risk group relative to the rest.
        Adjusts for covariates (e.g., genetic PCs) using logistic regression.
        Uses one-hot encoding of prediction quantiles to model risk enrichment in high-score groups.
        Returns either the top-bin odds ratio or all group-specific odds ratios.

    """
    
    assert num_classes in [
        1,
        2,
        None,
    ], "Only binary classification is supported by OR func now."

    if weights is None:
        weights = torch.ones_like(preds)

    if preds.sum() == 0:
        logger.warning("Cannot compute odds ratios for all zero predictions.")
        return torch.tensor([np.nan], device=preds.device)
    n_groups = 100 / top_x
    if top:
        relevant_bin = -1
    else:
        relevant_bin = 0

    if round(n_groups) != n_groups:
        raise ValueError(
            f"adjusted odds ratio must cleanly divide 100 ({top_x} does not"
        )
    n_groups = int(n_groups)

    bins = torch.quantile(
        preds,
        torch.linspace(0, 1, n_groups + 1, dtype=preds.dtype, device=preds.device)[
            1:-1
        ],
    )
    score_groups = torch.bucketize(preds, bins)
    if sum(abs(score_groups)) == 0:
        logger.warning(
            "Predictions all fall within the same score group, cannot compute odds ratio, perhaps increase the number of groups."
        )
        return torch.tensor([np.nan], device=preds.device)
    one_hot_score_groups = (
        torch.nn.functional.one_hot(score_groups, num_classes=n_groups)
        .to(torch.float32)
        .squeeze()
    )
    X = torch.cat([covariates, one_hot_score_groups], axis=1)

    # ds = TensorDataset(X, target)
    # clf = TorchLogisticReg(nfeatures=X.shape[1]).to(X.device)
    # clf.fit(ds=ds, max_iterations=100)
    clf = LogisticRegression(
        max_iter=1000
    )  

    clf.fit(
        X.numpy(), target.numpy().flatten(), sample_weight=weights.numpy().flatten()
    )

    if return_all:
        return np.exp(clf.coef_[0, covariates.shape[1] :])
    else:
        return np.exp(clf.coef_[0, covariates.shape[1] :])[relevant_bin]

def topXincidence_compute(
    preds: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
    top_x: Optional[int] = 1,
    n_groups: Optional[int] = 100,
    top: Optional[bool] = True,
    return_all: Optional[bool] = False,
    num_classes: Optional[int] = 2,
    average: Optional[str] = None,
    task: Optional[str] = "binary",
) -> Tensor:
    """Computes Top X % incidence.
    Args:
        preds: predictions from model (logits or probabilities)
        target: Ground truth labels
        top_x: top x out of n_groups
        n_groups: number of groups / bins to split the predictions into
        top: whether to go from top or bottom

        unused:
        mode: 'multi class multi dim' or 'multi-label' or 'categorical'
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class.
            Should be set to ``None`` for binary problems
        average: Defines the reduction that is applied to the output:
        sample_weights: sample weights for each data point
        num_classes: accepted only to have a valid signature with other classes and for checks
    
    Explained: 
        Measures the observed incidence of the outcome in the top X% of predicted risk scores.
        Divides predictions into n_groups (e.g., 100), selects the top X% highest-risk group.
        Optionally returns incidence for all score bins. -> Plot with incidence on y axis and ranked groups on x axis.


    """
    assert num_classes in [
        1,
        2,
        None,
    ], "Only binary classification is supported by OR func now."
    #     if num_classes or pos_label or average or sample_weights:
    #         raise NotImplementedError
    if weights is None:
        weights = torch.ones_like(preds)

    bounds = torch.linspace(0, 1, n_groups + 1, dtype=preds.dtype, device=preds.device)[
        1:-1
    ]
    bins = torch.quantile(preds, bounds)
    score_groups = torch.bucketize(preds, bins)

    if not return_all:
        if top:
            selection = score_groups > (n_groups - 1) - top_x
        else:
            selection = (
                score_groups < (n_groups - 1) - top_x
            )  # this has to be 99 because the score groups go from (incl.) 0 - 99
        if len(target[selection]) == 0:
            return np.nan
        if weights is None:
            return target[selection].sum() / target[selection].shape[0]

        weighted_target_sum = (target[selection] * weights[selection]).sum()
        weighted_count = weights[selection].sum()
        return weighted_target_sum / weighted_count
    else:
        incidences = []
        for score_group in range(
            n_groups
        ): 
            selection = score_groups == score_group
            if len(target[selection]) == 0:
                incidences.append(
                    torch.tensor(np.nan, device=target.device).unsqueeze(0)
                )
            else:
                weighted_target_sum = (target[selection] * weights[selection]).sum()
                weighted_count = weights[selection].sum()
                incidence = (weighted_target_sum / weighted_count).unsqueeze(0)
                incidences.append(incidence)
        return torch.cat(incidences)



def main(args):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # prints to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # saves to a file
    file_handler = logging.FileHandler("training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # set the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    version = 'F3_new_v3_202512'
    # define the base string for output
    output_path = f"mixmil__{args.target_col}__{args.balancing}__{args.embedding_name}_embed__CVsplit_{args.cv_split_n}__seed{args.seed}__Nepochs{args.n_epochs}__batch_size{args.batch_size}__abs_counts_{args.celltype_ABS}__{version}"

    # These remain configurable at the CLI level in the future, but are kept
    # here to preserve the existing DNA Nexus workflow.
    sex_col = "sex"
    age_col = "age"
    bmi_col = "bmi"
    smoking_col = "smoking_status_numeric"
    covariates_LR = [age_col, sex_col, bmi_col, smoking_col]
    celltype_ABS = [x for x in args.celltype_ABS.split('__') if x and x.lower() != 'none']
    fixed_effects = [sex_col, bmi_col, age_col, smoking_col] + celltype_ABS
    scale_fixed_effects = [bmi_col, age_col] + celltype_ABS
    scale_y = False
    ############################################################################################
    ######### import files ####################################################################
    ############################################################################################
    adata = sc.read_h5ad(args.adata_path)
    disease_table = pd.read_csv(args.disease_table_path, sep = '\t')
    print(f"Dimension of disease table: {disease_table.shape}")
    print(f"Head of disease table: {disease_table.head()}")

    CV_splits_table = pd.read_csv(args.cv_table_path, index_col = 0)
    CV_splits_table.index.name = args.donor_col
    CV_splits_table = CV_splits_table.reset_index()
    ##################################################################################################
    ####### Train and Test IDs from table ###################################################
    ##################################################################################################
    # from the CV split tables: subset the columns for the trait of interest
    matching_column = f'{args.target_col}__{args.balancing}__cv_split{args.cv_split_n}'
    
    #print([col for col in CV_splits_table.columns if "OBESITY" in col])

    adata.obs[args.donor_col] = adata.obs[args.donor_col].astype(str)
    disease_table[args.donor_col] = disease_table[args.donor_col].astype(str)
    CV_splits_table[args.donor_col] = CV_splits_table[args.donor_col].astype(str)
    print(f"Matching column: {matching_column}")
    print(f"CV_split table head: {CV_splits_table.head()}")
    print(f"donor column used: {args.donor_col}")
    train_ids = CV_splits_table.loc[CV_splits_table[matching_column] == 'Train', args.donor_col].astype(str).tolist()   
    test_ids = CV_splits_table.loc[CV_splits_table[matching_column] == 'Test', args.donor_col].astype(str).tolist()
    print(f"train ids list from CV table:{train_ids}")
    print(f"train ids list from CV table:{test_ids}")
    
    print(f"adata obs eid: {adata.obs[args.donor_col]}")
    print(f"adata obs eid values: {adata.obs[args.donor_col].values}")
    print(f"adata obs index: {adata.obs.index}")
    print(f"adata.obs columns: {adata.obs.columns}")
    print(f"adata obs dimensions: {adata.obs.shape}")
    
    print("Number of donors in adata:", adata.obs[args.donor_col].nunique())
    print("Number of donors in disease table:", disease_table[args.donor_col].nunique())
    print("Overlap:", len(set(adata.obs[args.donor_col]) & set(disease_table[args.donor_col])))

    
    adata_to_process = adata[adata.obs[args.donor_col].isin(train_ids + test_ids)].copy()
    print(f"adata to process dim:{adata_to_process.obs.shape}")
    adata_to_process.obs["split"] = np.where(adata_to_process.obs[args.donor_col].isin(train_ids), "train", "test").copy()
    print(f"adata to process obs split: {adata_to_process.obs['split'].value_counts()}")
    disease_table[args.donor_col] = disease_table[args.donor_col].astype(str)
    disease_dict = disease_table.set_index(args.donor_col)[args.target_col].to_dict()
    adata_to_process.obs[args.target_col] = adata_to_process.obs[args.donor_col].map(disease_dict)
    print(f"Value counts of the disease_col (there should be no -1 values: {adata_to_process.obs[args.target_col].value_counts(dropna=False)}")
    
    adata_to_process = adata_to_process[adata_to_process.obs[args.target_col].isin([1, 0])].copy()
    del adata
    gc.collect()

    ##################################################################################################
    ####### MixMIL ###################################################
    ##################################################################################################
    if sex_col in adata_to_process.obs.columns and not pd.api.types.is_numeric_dtype(adata_to_process.obs[sex_col]):
        mapped_sex = adata_to_process.obs[sex_col].map({'Female': 0, 'Male': 1})
        if mapped_sex.notna().all():
            adata_to_process.obs[sex_col] = mapped_sex.astype(np.float32)

    missing_fixed = [column for column in fixed_effects if column not in adata_to_process.obs.columns]
    if missing_fixed:
        raise KeyError(f"Missing fixed-effect columns in AnnData: {missing_fixed}")

    prepared = prepare_mixmil_anndata(
        adata_to_process,
        bag_key=args.donor_col,
        target_key=args.target_col,
        feature_key=args.embedding_name,
        feature_source='obsm',
        fixed_effects=fixed_effects,
        split_key='split',
        scale_fixed_effects=scale_fixed_effects,
        scale_target=scale_y,
    )
    if prepared.test is None:
        raise ValueError('The AnnData object does not contain any test bags')

    train_data = prepared.train
    test_data = prepared.test
    Xs, F_scaled, Y_train = train_data.Xs, train_data.F, train_data.Y
    test_Xs, F_scaled_test, test_Y = test_data.Xs, test_data.F, test_data.Y
    train_bags, test_bags = train_data.bag_ids, test_data.bag_ids
    train_cellid, test_cellid = train_data.cell_ids, test_data.cell_ids
    F_train_scaling_params = prepared.scaling_params
    Q, K = prepared.n_features, prepared.n_fixed_effects
    Y = Y_train
    mean, std = prepared.target_scaling if prepared.target_scaling is not None else (None, None)

    all_metric_rows = []
    
    try:

        # initialize MixMIL
        model = MixMIL.init_with_mean_model(Xs, F_scaled, Y_train, likelihood="binomial", n_trials=1)
        y_pred_init = model.predict(test_Xs)

        ### Train MixMIL
        device = "cuda:0"
        model, Xs_cuda, F_cuda, Y_cuda, test_Xs_cuda, test_Y_cuda = to_device((model, Xs, F_scaled, Y_train, test_Xs,Y_test), device)
        start_time = time.time()  
        history = model.fit(Xs_cuda, F_cuda, Y_cuda, 
                              n_epochs=args.n_epochs, 
                              batch_size=args.batch_size)
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_path + '__training_history.csv')
        # >>> history_df.columns 
        #Index(['loss', 'll', 'kld', 'epoch', 'step'], dtype='object')

        end_time = time.time()
        execution_time = end_time - start_time
        exec_time = round(execution_time/60, 2)
        print(f"Execution time in minutes: {exec_time}")
        
        row = {
            'CV_fold': args.cv_split_n,
            'Seed_value': args.seed,
            'N_epochs': args.n_epochs,
            'Training_time_minutes': exec_time,
            'n_donor_train': len(Y_train),
            'n_donor_test': len(Y_test),
            'n_cases_test': int((Y_test==1).sum()),
            'n_controls_test': int((Y_test==0).sum()),
        }
        
        all_metric_rows.append(row)

        model.to("cpu")
        test_Xs = [x.cpu() for x in test_Xs_cuda]
        #####################################################################################
        ###################### MixMIL_trained_emb_only ###########################################
        ########################################################################################
        y_pred_logits = model.predict(test_Xs).cpu()
        #####################################################################################
        ####################### with embeddings and fixed effects ######################
        ########################################################################################
        beta_u, beta_z = model.get_betas()
        # y_pred = (F_scaled_test @ model.alpha + test_Xs @ beta_z).cpu().numpy()
        y_pred_logits_FE = model.predict_logits(test_Xs, F_scaled_test).detach().cpu().numpy()
        #################################################################################################
        # get the true Y for the testing set:
        Y_test_after = [y.cpu() for y in test_Y_cuda]
        y_test_np = np.array([tensor.item() for tensor in Y_test_after]) # TRUE

        ###### save a table with:
        df = pd.DataFrame({
                                "donor_id": test_bags,
                                "pred_logit_init_mean_MixMIL": y_pred_init.squeeze().tolist(),
                                "pred_logit_MixMIL_trained_emb_only": y_pred_logits.squeeze().tolist(),
                                "pred_logit_MixMIL_trained_emb_FE": y_pred_logits_FE.squeeze().tolist(),
                                "true_label": y_test_np.flatten().tolist()
                            })
        print("Check true label column")
        print(df)
        # save cell level metrics
        attw_beforeSM = torch.cat(model.get_weights(test_Xs)[1], dim=0).numpy()
        attw_afterSM = torch.cat(model.get_weights(test_Xs)[0], dim=0).numpy()

        # add donor eid as a column (use a dictionary like {'cell_id': 'eid'}
        cellid_eid_dict = dict(zip(adata_to_process.obs.index.astype(str), adata_to_process.obs[args.donor_col].astype(str)))
        
        with open(output_path+"__eid_cellid_dict_test_set.pkl", "wb") as f:
            pickle.dump(cellid_eid_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # dictionary with cell level metrics
        
        dict_test = {
                     'cellid_test': test_cellid, 
                     'attw_beforeSM': attw_beforeSM,
                     'attw_afterSM': attw_afterSM
                    }

        with open(output_path+"__cellid_dict_attw_test_set.pkl", "wb") as f:
            pickle.dump(dict_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        
            
        ##################################################################################################
        ####### Logistic regression  ###################################################
        ##################################################################################################
        train_emb = [X.mean(dim=0).numpy() for X in Xs]  
        train_emb = np.vstack(train_emb)
        train_cov = F_scaled.cpu().numpy()

        test_emb = [X.mean(dim=0).numpy() for X in test_Xs]  
        test_emb = np.vstack(test_emb)
        test_cov = F_scaled_test.cpu().numpy()

        y_train_LR = Y_train.cpu().numpy().ravel()
        y_test_LR = Y_test.cpu().numpy().ravel()

        # covariates only
        X_train_cov = train_cov                     
        X_test_cov  = test_cov                     

        # covariates + mean embedding
        X_train_cov_emb = np.hstack([train_cov, train_emb])  
        X_test_cov_emb  = np.hstack([test_cov, test_emb])    

        y_pred_NULL, y_pred_proba_NULL = train_and_predict(X_train_cov, X_test_cov, y_train_LR, y_test_LR, 
                                                           "Covariates only (NULL)")
        y_pred_LR, y_pred_proba_LR = train_and_predict(X_train_cov_emb, X_test_cov_emb, y_train_LR, y_test_LR, 
                                                       "Covariates + embedding (LR)")


        df_NULL_LR = pd.DataFrame({
            "donor_id": test_bags,
            "pred_NULL": y_pred_NULL,
            "pred_proba_NULL": y_pred_proba_NULL[:, 1], 
            "pred_LR": y_pred_LR,
            "pred_proba_LR": y_pred_proba_LR[:, 1]
                                        })
        
        predictions = df.merge(df_NULL_LR, on="donor_id", how="inner")
        
        # --- SAFETY CHECKS ---
        assert len(test_bags) == len(y_pred_logits) == len(y_test_np) == len(y_pred_LR), "Length mismatch between donors, predictions, LR predictions and true labels"
        missing = set(test_bags) - set(df_NULL_LR['donor_id'])
        if missing:
            logger.warning(f"{len(missing)} test donors missing from LR outputs: {list(sorted(missing))[:5]} ...")
    
    
        
        # add covariates to the predictions table (age, sex, bmi, smoking status)
        #covariates_LR = ["age", "sex", "bmi", "smoking_status_numeric"]
        
        for col in covariates_LR:
            col_dict = adata_to_process.obs.set_index(args.donor_col)[col].to_dict()
            predictions[col] = predictions['donor_id'].map(col_dict)
        

        predictions.to_csv(output_path + "__predictions.csv")
        ########################################################################################
        ###################### save MixMIL weights ############################################
        ########################################################################################
        torch.save(model.state_dict(), output_path + '__mixmil_trained_model.pt')

        # Write a scverse-native result object as well as the legacy tables.
        # Attention is aligned by cell ID; donor predictions remain donor-level
        # metadata in .uns instead of being broadcast into every cell.
        annotated = add_attention_to_anndata(
            adata_to_process,
            train_data,
            model.get_weights(train_data.Xs)[0],
            key='mixmil_attention',
        )
        annotated = add_attention_to_anndata(
            annotated,
            test_data,
            model.get_weights(test_data.Xs)[0],
            key='mixmil_attention',
            inplace=True,
        )
        annotated = add_bag_predictions_to_anndata(
            annotated,
            test_bags,
            y_pred_logits_FE,
            key='mixmil',
            inplace=True,
        )
        annotated.write_h5ad(output_path + '__annotated.h5ad')

    except Exception as e:
        logging.exception(f"Error during training for CV step {args.cv_split_n}")
        print(f"Raised error: {e}")
        raise
    finally: 
        print(f'Finished training')

    training_time_df = pd.DataFrame(all_metric_rows)
    training_time_df.to_csv(output_path + "__training_time_and_support.csv")        

    print(f"Model successfully run with number of epochs = {args.n_epochs}")
    print(f"Model successfully run with cv_split_n = {args.cv_split_n}")
        
    # save convergence plot 
    history_avg_df = history_df.groupby('epoch').agg('mean').reset_index()

    hist_cols = ['loss', 'll', 'kld']
  
    # from Jan:
    history_avg_df
    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    for ax, col in zip(axs, hist_cols):
        ax.plot(history_avg_df["epoch"], history_avg_df[col])
        ax.set_title(col)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path + "__convergence_plot.pdf", format="pdf", bbox_inches="tight")


    # prediction performance metrics: 

    summary_rows = []

    pred_prob_init = torch.sigmoid(torch.tensor(predictions["pred_logit_init_mean_MixMIL"]))  # PROBABILITY
    pred_class_init = torch.round(pred_prob_init)  # CLASS
        
    pred_prob_train = torch.sigmoid(torch.tensor(predictions["pred_logit_MixMIL_trained_emb_only"]))  # PROBABILITY
    pred_class_train = torch.round(pred_prob_train)  # CLASS
        
    pred_prob_train_FE = torch.sigmoid(torch.tensor(predictions["pred_logit_MixMIL_trained_emb_FE"]))  # PROBABILITY
    pred_class_train_FE = torch.round(pred_prob_train_FE)     # CLASS
    # for AUCROC and AUPRC you need the true label + the predicted PROB (not the logit)
    AUROC_null = roc_auc_score(predictions['true_label'], predictions['pred_proba_NULL'])
    AUROC_LR = roc_auc_score(predictions['true_label'], predictions['pred_proba_LR'])
    AUROC_baseline = roc_auc_score(predictions['true_label'], pred_prob_init)
    AUROC_trained = roc_auc_score(predictions['true_label'], pred_prob_train)
    AUROC_trained_FE = roc_auc_score(predictions['true_label'], pred_prob_train_FE)
    AUPRC_null = average_precision_score(predictions['true_label'], predictions['pred_proba_NULL'])
    AUPRC_LR = average_precision_score(predictions['true_label'], predictions['pred_proba_LR'])
    AUPRC_baseline = average_precision_score(predictions['true_label'], pred_prob_init)
    AUPRC_trained = average_precision_score(predictions['true_label'], pred_prob_train)
    AUPRC_trained_FE = average_precision_score(predictions['true_label'], pred_prob_train_FE)
    # for f1 score you need true vs predicted label
    f1_score_null = f1_score(predictions['true_label'], predictions['pred_NULL'])
    f1_score_LR = f1_score(predictions['true_label'], predictions['pred_LR'])
    f1_score_baseline = f1_score(predictions['true_label'], pred_class_init)
    f1_score_trained = f1_score(predictions['true_label'], pred_class_train)
    f1_score_trained_FE = f1_score(predictions['true_label'], pred_class_train_FE) 

    summary_rows.append({
           "embedding": args.embedding_name,
            "trait": args.target_col, 
            "n_epochs": args.n_epochs,
            "cv_fold": args.cv_split_n,
            "seed": args.seed,
            "f1_score_null": f1_score_null,
            "f1_score_LR": f1_score_LR,
            "f1_score_baseline": f1_score_baseline,
            "f1_score_trained": f1_score_trained,
            "f1_score_trained_FE": f1_score_trained_FE,
            "AUROC_null": AUROC_null,
            "AUROC_LR": AUROC_LR,
            "AUROC_baseline": AUROC_baseline,
            "AUROC_trained": AUROC_trained,
            "AUROC_trained_FE": AUROC_trained_FE,
            "AUPRC_null": AUPRC_null,
            "AUPRC_LR": AUPRC_LR,
            "AUPRC_baseline": AUPRC_baseline, 
            "AUPRC_trained": AUPRC_trained, 
            "AUPRC_trained_FE":AUPRC_trained_FE 
        })

    ##### add top X incidence: 
    top_x_list = [1, 2]
    n_groups = 10
    top = True
    pred_cols = [
    "pred_logit_init_mean_MixMIL",
    "pred_logit_MixMIL_trained_emb_only",
    "pred_logit_MixMIL_trained_emb_FE",
    "pred_proba_NULL",
    "pred_proba_LR"]

    for top_x in top_x_list:
        for col in pred_cols:
            topX_incidence = topXincidence_compute(preds = torch.tensor(predictions[col]),  # logits or probabilities
                                   target = torch.tensor(predictions['true_label']),  # ground truth labels
                                   top_x = top_x, # top x out of n groups? 
                                   n_groups = n_groups,
                                   top = True
                              ).numpy().item()
            new_col_name = f"top{top_x}_incidence_in_{n_groups}_groups__{col}"
            summary_rows[0][new_col_name] = topX_incidence

    summary_df = pd.DataFrame(summary_rows)
    
    ##### here add the topx odds ratio from Lucas 
    ##### TODO
    #############################################
    summary_df.to_csv(output_path + "__performance_metrics.csv")
    
    ### Incidence plots:
    with PdfPages(output_path + "__incidence_plots.pdf") as pdf:
        n_groups_list = [10, 100]
        for n_groups in n_groups_list:
            for col in pred_cols:
                incidences = topXincidence_compute(
                                       preds = torch.tensor(predictions[col]),  # logits or probabilities
                                       target = torch.tensor(predictions['true_label']),  # ground truth labels
                                       n_groups = n_groups,
                                       return_all=True
                                   )
                incidences_np = incidences.cpu().numpy()

                plt.figure(figsize=(7,4))
                plt.plot(range(1, len(incidences_np)+1), incidences_np,
                         marker=".", 
                         linewidth=1)
                plt.xlabel(f"Risk score bin (ranked), number of groups: {n_groups}")
                plt.ylabel("Incidence")
                plt.title(f"Incidence in {n_groups} prediction bins for column: {col}")
                plt.grid(alpha=0.3)
                pdf.savefig()
                plt.close()  
    
    ############################
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    parser.add_argument("--cv_split_n", required=True, type=int, help="number of cv split: choice between 0, 1, 2")
    parser.add_argument("--balancing", required=True, type=str,help="whether or not to balance classes: one between 'balanced' or 'unbalanced'")
    parser.add_argument("--target_col", required=True, type=str,help="name of target column (disease)")
    parser.add_argument("--donor_col", required=False, default = 'eid',
                        type=str,help="name of the donor id column (default: 'eid')")
    parser.add_argument("--embedding_name", required=True, type=str, 
                        help="embedding type is scanvi for Freeze3 - you also chose the std method: one between 'scanvi', 'scanvi_pca_10', 'scanvi_pca_20', 'scanvi_pca_30', 'scanvi_scaled'")
    parser.add_argument("--cv_table_path", required=True, type=str, help="Path for the cross validation information")
    parser.add_argument("--seed", required=True, type=int, help="Seed of choice")
    parser.add_argument("--n_epochs", required=True, type=int, help="Number of epochs for the training")
    parser.add_argument("--adata_path", required=True, type=str, help="Path for the input anndata")
    parser.add_argument("--disease_table_path", required=True, type=str, help="Path for the disease table")
    #parser.add_argument("--celltype_ABS", required=True, type=str, help="Celltype to be used as additional fixed effect")
    parser.add_argument("--batch_size", required=False, default=64, type=int, help="Batch size for training")
    parser.add_argument(
        "--celltype_ABS",
        required=False,
        default="none",
        type=str,
        help="Double-underscore-separated cell-count columns to add as fixed effects",
    )
    args = parser.parse_args()
    main(args)
