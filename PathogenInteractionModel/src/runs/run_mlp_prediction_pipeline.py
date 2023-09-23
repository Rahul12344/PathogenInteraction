import os, sys
import pandas as pd
import numpy as np

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from utils.train_test_split import train_test_split_keys, get_text_data
from training.train_model import train_ngram_model
from utils.metrics import auroc_roc_curve, auprc_prc_curve, get_confusion_matrix, plot_auroc_roc_curve, plot_auprc_prc_curve, plot_confusion_matrix

if __name__ == '__main__':
    # load config from config.yaml
    c = config()
    
    """
    Step 1: Convert text to ngrams and vectorize
    Step 2: Train model on the vectorized data
    Step 3: Plot metrics
    """
    
    # randomly select negatives to encode
    df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))
    positives = []
    positives_datasets = []
    negatives = []
    negatives_datasets = []
    for dataset in c['train_datasets']:
        positives += df[(df['label'] == 1) & (df['dataset'].str.contains(dataset))]['file_name'].tolist()
        positives_datasets += [dataset]*len(positives)
        if c['balance_dataset']:
            negatives += df[(df['label'] == 0) & (df['dataset'].str.contains(dataset))].sample(n=len(positives), random_state=c['random_state'])['file_name'].tolist()
        else:
            negatives += df[(df['label'] == 0) & (df['dataset'].str.contains(dataset))].sample(n=int(c['ratio']*len(positives)), random_state=c['random_state'])['file_name'].tolist()
        negatives_datasets += [dataset]*len(negatives)
    positive_labels = [1 for _ in positives]
    negative_labels = [0 for _ in negatives]

    filenames = positives + negatives
    labels = positive_labels + negative_labels
    datasets = positives_datasets + negatives_datasets
    
    # split the data into train, val, and test
    X_train_keys, X_val_keys, X_test_keys, _, _, _ = train_test_split_keys(c, filenames, labels)
    X_train, X_val, X_test, y_train, y_val, y_test = get_text_data(c, X_train_keys, X_val_keys, X_test_keys, positives, negatives, positives_datasets, negatives_datasets) 
    data = ((X_train,y_train),(X_val,y_val),(X_test,y_test))
    
    # set up model and trainer
    model, predicted_test, _, _ = train_ngram_model(c, data)

    # calculate AUROC and AUPRC and plot metrics
    fpr, tpr, roc_auc = auroc_roc_curve(y_test, predicted_test)
    plot_auroc_roc_curve(c, fpr, tpr, roc_auc, c['train_datasets'], c['test_datasets'], model_type='n_gram_MLP')
    
    precision, recall, prc_auc = auprc_prc_curve(y_test, predicted_test)
    plot_auprc_prc_curve(c, precision, recall, prc_auc, c['train_datasets'], c['test_datasets'], model_type='n_gram_MLP')
    
    cfm = get_confusion_matrix(y_test, predicted_test)
    plot_confusion_matrix(c, cfm, c['train_datasets'], c['test_datasets'], percentage=True, model_type='n_gram_MLP')
    plot_confusion_matrix(c, cfm, c['train_datasets'], c['test_datasets'], percentage=False, model_type='n_gram_MLP')