import os, sys
import torch
import pandas as pd
import numpy as np

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from embed.embedding_models import load_bert_model_from_pretrained
from embed.embedding_pipeline import embedding_pipeline
from utils.train_test_split import train_test_split_keys, get_tensors_to_train, get_tensors_to_test, get_text_data, text_data
from training.datasets import BERTDataset, get_dataset_loaders, get_LOO_dataset_loader
from training.embedding_models import LogisticRegression
from training.train_model import TrainModel, train_ngram_model
from training.cost_functions import ElasticNetBCELoss
from utils.metrics import auroc_roc_curve
from utils.metrics import auprc_prc_curve
from utils.metrics import get_confusion_matrix
from utils.metrics import plot_auroc_roc_curve
from utils.metrics import plot_auprc_prc_curve
from utils.metrics import plot_confusion_matrix
from utils.metrics import plot_training_data_pca
from utils.metrics import plot_acc_and_loss
from utils.metrics import create_dot_plot_of_transfer_learning_performance

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # load config from config.yaml
    c = config()
    results = []


    # run the prediction pipeline
    
    train_datasets = [['virus'], ['bacteria'], ['malaria']]
    test_datasets = [['virus'], ['bacteria'], ['malaria']]
    # load the embeddings
    
    # iterate through all combinations of train and test datasets
    if c['run_all_combinations']:
        for train_dataset in train_datasets:
            for test_dataset in test_datasets:
                
                if c['model'] != 'mlp':
                    positive_tensors = {}
                    negative_tensors = {}
                    for dataset in train_dataset:
                        positive_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt')))
                        negative_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt')))
                    
                    # get file names from torch tensors and assign labels
                    positive_filenames = [k for k, _ in positive_tensors.items()]
                    positive_labels = [1 for k in positive_filenames]
                    if c['balance_dataset']:
                        negative_filenames = [k for k, _ in negative_tensors.items()][0:len(positive_tensors)]
                    negative_labels = [0 for k in negative_filenames]
                    
                    filenames = positive_filenames + negative_filenames
                    labels = positive_labels + negative_labels
                    
                    # split the train set data into train, val, and test
                    X_train_keys, X_val_keys, X_test_keys, _, _, _ = train_test_split_keys(c, filenames, labels)
                    X_train, X_val, X_test, y_train, y_val, y_test = get_tensors_to_train(X_train_keys, X_val_keys, X_test_keys, positive_tensors, negative_tensors) 
                    
                    # plot the training data in 2D
                    plot_training_data_pca(c=c, 
                                        X_train=X_train, 
                                        y_train=y_train, 
                                        train_dataset=train_dataset)
                    
                    # set up dataset and dataloader
                    train_set = BERTDataset(c, device, X_train, y_train)
                    val_set = BERTDataset(c, device, X_val, y_val)
                    test_set = BERTDataset(c, device, X_test, y_test)
                    
                    train_loader, val_loader, test_loader = get_dataset_loaders(train_set, val_set, test_set, c['batch_size'])    
                    
                    # set up model and trainer
                    classifier = LogisticRegression(c=c, device=device)
                    
                    trainer = TrainModel(c=c, 
                                        classifier=classifier, 
                                        epochs=c['epochs'], 
                                        lr=c['learning_rate'],
                                        weight_decay=c['weight_decay'],
                                        min_delta=c['min_delta'],
                                        tolerance=c['tolerance'])
                    
                    cost = ElasticNetBCELoss(c=c, 
                                            device=device, 
                                            alpha=c['alpha'], 
                                            beta=c['beta'])
                    
                    # fit training data and validate
                    train_acc_history, train_loss_history, val_acc_history, val_loss_history = trainer.fit(cost, train_loader, val_loader)
                    predicted_test, test_acc, test_loss = trainer.predict(test_loader, cost)
                    y_test = y_test.cpu().detach().numpy()
                
                if c['model'] == 'mlp':
                    df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))
                    positives = []
                    positives_datasets = []
                    negatives = []
                    negatives_datasets = []
                    for dataset in train_dataset:
                        positives += df[(df['label'] == 1) & (df['dataset'] == dataset)]['file_name'].tolist()
                        positives_datasets += [dataset]*len(positives)
                        if c['balance_dataset']:
                            negatives += df[(df['label'] == 0) & (df['dataset'] == dataset)].sample(n=len(positives), random_state=c['random_state'])['file_name'].tolist()
                        else:
                            negatives += df[(df['label'] == 0) & (df['dataset'] == dataset)].sample(n=int(c['ratio']*len(positives)), random_state=c['random_state'])['file_name'].tolist()
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
                    model, predicted_test, vectorizer, selector = train_ngram_model(c, data)                    
                    
                

                # plot metrics
                """plot_acc_and_loss(c=c, 
                                train_acc_history=train_acc_history, 
                                train_loss_history=train_loss_history,
                                val_acc_history=val_acc_history, 
                                val_loss_history=val_loss_history, 
                                test_acc=test_acc, 
                                test_loss=test_loss, 
                                train_dataset=train_dataset,
                                test_dataset=train_dataset,
                                model_type=c['model'])"""

                # calculate AUROC and AUPRC and plot metrics
                fpr, tpr, roc_auc = auroc_roc_curve(y_test, predicted_test)
                plot_auroc_roc_curve(c=c, 
                                    fpr=fpr, 
                                    tpr=tpr, 
                                    roc_auc=roc_auc, 
                                    train_dataset=train_dataset, 
                                    test_dataset=train_dataset,
                                    model_type=c['model'])
                
                precision, recall, prc_auc = auprc_prc_curve(y_test, predicted_test)
                plot_auprc_prc_curve(c=c, 
                                    precision=precision, 
                                    recall=recall, 
                                    prc_auc=prc_auc, 
                                    train_dataset=train_dataset, 
                                    test_dataset=train_dataset,
                                    model_type=c['model'])
                
                cfm = get_confusion_matrix(y_test, predicted_test)
                plot_confusion_matrix(c=c, 
                                    confusion_matrix=cfm, 
                                    train_dataset=train_dataset, 
                                    test_dataset=train_dataset,
                                    model_type=c['model'], 
                                    percentage=True)
                plot_confusion_matrix(c=c, 
                                    confusion_matrix=cfm, 
                                    train_dataset=train_dataset, 
                                    test_dataset=train_dataset,
                                    model_type=c['model'], 
                                    percentage=False)
                
                # load test data
                
                if c['model'] != 'mlp':
                    positive_test_tensors = {}
                    negative_test_tensors = {}
                    for dataset in test_dataset:
                        positive_test_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt')))
                        negative_test_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt')))
                        
                    LOOX_test, LOOy_test = get_tensors_to_test(positive_test_tensors, negative_test_tensors)
                    LOO_test_set = BERTDataset(c, device, LOOX_test, LOOy_test)
                    LOO_loader = get_LOO_dataset_loader(LOO_test_set, c['batch_size'])    
                    
                    # evaluate on test data
                    LOO_predicted_test, LOO_test_acc, LOO_test_loss = trainer.predict(LOO_loader, cost)
                    LOOy_test = LOOy_test.cpu().detach().numpy()
                else:
                    positive_filenames = []
                    negative_filenames = []
                    for dataset in test_dataset:
                        df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))
                        positive_filenames += df[(df['label'] == 1) & (df['dataset'] == dataset)]['file_name'].tolist()
                        negative_filenames += df[(df['label'] == 0) & (df['dataset'] == dataset)]['file_name'].tolist()
                    labels = [1 for _ in positive_filenames] + [0 for _ in negative_filenames]
                    positive_datasets = [dataset] * len(positive_filenames)
                    negative_datasets = [dataset] * len(negative_filenames)
                    
                    filenames = positive_filenames + negative_filenames
                    
                    X, LOOy_test = text_data(c, positive_filenames, negative_filenames, positive_datasets, negative_datasets)
                    LOO_test = vectorizer.transform(X)
                    LOO_test = selector.transform(LOO_test).astype('float32').todense()
                    
                    LOO_predicted_test = model.predict(LOO_test)
                
                
                # plot LOO metrics
                
                # calculate AUROC and AUPRC and plot metrics
                fpr, tpr, test_roc_auc = auroc_roc_curve(LOOy_test, LOO_predicted_test)
                plot_auroc_roc_curve(c=c, 
                                    fpr=fpr, 
                                    tpr=tpr, 
                                    roc_auc=test_roc_auc, 
                                    train_dataset=train_dataset, 
                                    test_dataset=test_dataset, 
                                    model_type=c['model'])
                
                precision, recall, test_prc_auc = auprc_prc_curve(LOOy_test, LOO_predicted_test)
                plot_auprc_prc_curve(c=c, 
                                    precision=precision, 
                                    recall=recall, 
                                    prc_auc=test_prc_auc, 
                                    train_dataset=train_dataset, 
                                    test_dataset=test_dataset, 
                                    model_type=c['model'])
                
                cfm = get_confusion_matrix(LOOy_test, LOO_predicted_test)
                plot_confusion_matrix(c=c, 
                                    confusion_matrix=cfm, 
                                    train_dataset=train_dataset, 
                                    test_dataset=test_dataset, 
                                    model_type=c['model'], 
                                    percentage=True)
                plot_confusion_matrix(c=c, 
                                    confusion_matrix=cfm, 
                                    train_dataset=train_dataset, 
                                    test_dataset=test_dataset, 
                                    model_type=c['model'], 
                                    percentage=False)
                
                """plot_acc_and_loss(c=c, 
                                train_acc_history=train_acc_history, 
                                train_loss_history=train_loss_history,
                                val_acc_history=val_acc_history, 
                                val_loss_history=val_loss_history, 
                                test_acc=LOO_test_acc, 
                                test_loss=LOO_test_loss, 
                                train_dataset=train_dataset,
                                test_dataset=test_dataset,
                                model_type=c['model'])"""
                
                results.append(['_'.join(train_dataset)+':'+'_'.join(test_dataset), '_'.join(train_dataset), '_'.join(test_dataset), roc_auc, prc_auc, test_roc_auc, test_prc_auc])
                LOO_predicted_test = np.array(LOO_predicted_test).flatten()
                df = pd.DataFrame({'file_name':filenames, 'label':labels, 'dataset': '_'.join(train_dataset)+':'+'_'.join(test_dataset), 'prediction':LOO_predicted_test, 'rounded_prediction':LOO_predicted_test.round()})
                df.to_csv(os.path.join(c['metrics_dir'], f'{c["model"]}_{"_".join(train_dataset)+":"+"_".join(test_dataset)}_transfer_predictions.csv'))
                
            # save results
            df = pd.DataFrame(results, columns=['dataset', 'train_dataset', 'test_dataset', 'same_set_roc_auc', 'same_set_prc_auc', 'test_roc_auc', 'test_prc_auc'])
            df.to_csv(os.path.join(c['metrics_dir'], f'{c["model"]}_transfer_results.csv'))
    
    df = pd.read_csv(os.path.join(c['metrics_dir'], f'{c["model"]}_transfer_results.csv'))
    create_dot_plot_of_transfer_learning_performance(c, df)
         
    
    
