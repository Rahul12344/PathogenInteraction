import os, sys
import torch
import pandas as pd
import numpy as np

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from utils.train_test_split import train_test_split_keys, get_tensors_to_train, get_text_data
from training.datasets import BERTDataset, get_dataset_loaders
from training.embedding_models import LogisticRegression
from training.train_model import TrainModel, train_ngram_model, cross_validate_torch_model
from training.cost_functions import RegressionLoss
from utils.metrics import auroc_roc_curve, auprc_prc_curve, plot_auroc_roc_curve_comp, plot_auprc_prc_curve_comp

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# run this pipeline after generating embeddings
if __name__ == '__main__':
    # load config from config.yaml
    c = config()

    # run the prediction pipeline
    
    # load the embeddings
    positive_tensors = {}
    negative_tensors = {}
    for dataset in c['train_datasets']:
        positive_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt')))
        negative_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt')))
    
    # get file names from torch tensors and assign labels
    positive_filenames = [k for k, _ in positive_tensors.items()]
    positive_labels = [1 for _ in positive_filenames]
    if c['balance_dataset']:
        negative_filenames = [k for k, _ in negative_tensors.items()][0:len(positive_tensors)]
    else:
        negative_filenames = [k for k, _ in negative_tensors.items()][0:c['ratio']*len(positive_tensors)]
    negative_labels = [0 for _ in negative_filenames]
    
    filenames = positive_filenames + negative_filenames
    labels = positive_labels + negative_labels
    
    # split the data into train, val, and test
    X_train_keys, X_val_keys, X_test_keys, _, _, _ = train_test_split_keys(c, filenames, labels)
    X_train, X_val, X_test, y_train, y_val, y_test = get_tensors_to_train(X_train_keys, X_val_keys, X_test_keys, positive_tensors, negative_tensors) 
    
    # set up dataset and dataloader
    train_set = BERTDataset(c, device, X_train, y_train)
    val_set = BERTDataset(c, device, X_val, y_val)
    test_set = BERTDataset(c, device, X_test, y_test)
    
    train_loader, val_loader, test_loader = get_dataset_loaders(train_set, val_set, test_set, c['batch_size'])    
    
    # set up model and trainer
    classifier = LogisticRegression(c=c, device=device)
    trainer = TrainModel(c=c, classifier=classifier)
    cost = RegressionLoss(c=c, device=device)
    
    # fit training data and validate
    train_acc_history, train_loss_history, val_acc_history, val_loss_history = trainer.fit(cost, train_loader, val_loader)
    predicted_test, test_acc, test_loss = trainer.predict(test_loader, cost)
    y_test = y_test.cpu().detach().numpy()

    # calculate AUROC and AUPRC and plot metrics
    blue_fpr, blue_tpr, blue_roc_auc = auroc_roc_curve(y_test, predicted_test)
    blue_precision, blue_recall, blue_prc_auc = auprc_prc_curve(y_test, predicted_test)

    # randomly select negatives to encode
    df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))
    
    positive_datasets = []
    negative_datasets = []
    for dataset in c['train_datasets']:
        positives = df[(df['label'] == 1) & (df['dataset'].str.contains(c['dataset']))]['file_name'].tolist()
        positive_datasets.extend([c['dataset']]*len(positives))
        if not c['balanced_dataset']:
            negatives = df[(df['label'] == 0) & (df['dataset'].str.contains(c['dataset']))].sample(n=int(c['ratio']*len(positives)), random_state=c['random_state'])['file_name'].tolist()
        else:
            negatives = df[(df['label'] == 0) & (df['dataset'].str.contains(c['dataset']))].sample(n=len(positives), random_state=c['random_state'])['file_name'].tolist()
        negative_datasets.append([c['dataset']]*len(negatives))
    else:
        positives = []
        negatives = []
        for dataset in c['combined-datasets-list']:
            positives += df[(df['label'] == 1) & (df['dataset'].str.contains(dataset))]['file_name'].tolist()
            positive_datasets.append(dataset)
            negatives += df[(df['label'] == 0) & (df['dataset'].str.contains(dataset))].sample(n=int(c['ratio']*len(positives)), random_state=c['random_state'])['file_name'].tolist()
            negative_datasets.append(dataset)
    
    positive_labels = [1 for _ in positives]
    negative_labels = [0 for _ in negatives]

    filenames = positives + negatives
    labels = positive_labels + negative_labels
    
    # split the data into train, val, and test
    X_train_keys, X_val_keys, X_test_keys, _, _, _ = train_test_split_keys(c, filenames, labels)
    X_train, X_val, X_test, y_train, y_val, y_test = get_text_data(c, X_train_keys, X_val_keys, X_test_keys, positives, negatives) 
    data = ((X_train,y_train),(X_val,y_val),(X_test,y_test))
    
    # set up model and trainer
    model, predicted_test, _, _ = train_ngram_model(c, data)

    # calculate AUROC and AUPRC and plot metrics
    mlp_fpr, mlp_tpr, mlp_roc_auc = auroc_roc_curve(y_test, predicted_test)
    mlp_precision, mlp_recall, mlp_prc_auc = auprc_prc_curve(y_test, predicted_test)
    
    # plot AUROC and AUPRC
    plot_auroc_roc_curve_comp(c, c['dataset'], blue_fpr, blue_tpr, blue_roc_auc, mlp_fpr, mlp_tpr, mlp_roc_auc)
    plot_auprc_prc_curve_comp(c, c['dataset'], blue_precision, blue_recall, blue_prc_auc, mlp_precision, mlp_recall, mlp_prc_auc)
