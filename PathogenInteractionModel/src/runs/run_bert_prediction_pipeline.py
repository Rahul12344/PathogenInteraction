import os, sys
import torch
import pandas as pd
import numpy as np

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from embed.embedding_models import load_bert_model_from_pretrained
from embed.embedding_pipeline import embedding_pipeline
from utils.train_test_split import train_test_split_keys, get_tensors_to_train, get_tensors_to_test, get_positive_tensors
from training.datasets import BERTDataset, get_dataset_loaders, get_LOO_dataset_loader
from training.embedding_models import LogisticRegression
from training.train_model import TrainModel
from training.cost_functions import ElasticNetBCELoss
from utils.metrics import auroc_roc_curve, auprc_prc_curve, get_confusion_matrix, plot_auroc_roc_curve, plot_auprc_prc_curve, plot_confusion_matrix, plot_training_data_pca, plot_acc_and_loss

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # load config from config.yaml
    c = config()
    
    """
    Step 1: Load embedding model of choice
    Step 2: Generate embeddings for positives and negatives randomly selected
    Step 3: Train model on the embeddings
    Step 4: Plot metrics
    """
    
    # randomly select negatives to encode
    df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))
    
    # run the embedding pipeline
    if c['should_generate_embeddings']:
        # load BERT model
        if c['embedding_model'] == 'bluebert':
            tokenizer, model = load_bert_model_from_pretrained(c, device)
        # if combined datasets, run the embedding pipeline for each dataset; 
        # otherwise, run the embedding pipeline for the specified dataset
        for dataset in c['train_datasets']:
            positives = df[(df['label'] == 1) & (df['dataset'].str.contains(dataset))]['file_name'].tolist()
            
            # if not balanced, sample negatives;
            # otherwise, use ratio of negatives
            if not c['balanced_dataset']:
                negatives = df[(df['label'] == 0) & (df['dataset'].str.contains(dataset))].sample(n=int(c['ratio']*len(positives)), random_state=c['random_state'])['file_name'].tolist()
            else:
                negatives = df[(df['label'] == 0) & (df['dataset'].str.contains(dataset))].sample(n=len(positives), random_state=c['random_state'])['file_name'].tolist()
            embedding_pipeline(c, dataset, model, tokenizer, positives, negatives, device)
        
    # run the prediction pipeline
    
    # load the embeddings
 
    positive_tensors = {}
    negative_tensors = {}
    for dataset in c['train_datasets']:
        positive_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt')))
        negative_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt')))
    
    # get file names from torch tensors and assign labels
    
    positives_ratio = 0.8
    positive_filenames = [k for k, _ in positive_tensors.items()]
    if c['balance_dataset']:
        negative_filenames = [k for k, _ in negative_tensors.items()][0:len(positive_tensors)]
    negative_labels = [0 for k in negative_filenames]
    
    positive_filenames = [k for k, _ in positive_tensors.items()][0:int(positives_ratio*len(positive_tensors))]
    positive_labels = [1 for k in positive_filenames]
    remaining_positives_filenames = [k for k, _ in positive_tensors.items()][int(positives_ratio*len(positive_tensors)):]
    remaining_positive_labels = [1 for k in remaining_positives_filenames]
    
    filenames = positive_filenames + negative_filenames
    labels = positive_labels + negative_labels
    
    # split the train set data into train, val, and test
    X_train_keys, X_val_keys, X_test_keys, _, _, _ = train_test_split_keys(c, filenames, labels)
    count = 0
    for filename in X_train_keys:
        if filename in positive_filenames:
            count += 1
    print(count)
    print(len(positive_filenames + remaining_positives_filenames))
        
    X_train, X_val, X_test, y_train, y_val, y_test = get_tensors_to_train(X_train_keys, X_val_keys, X_test_keys, positive_tensors, negative_tensors) 
    
    X, y = get_positive_tensors(remaining_positives_filenames, positive_tensors)
    X_test = torch.cat((X_test, X), 0)
    y_test = torch.cat((y_test, y))
    
    # plot the training data in 2D
    plot_training_data_pca(c=c, 
                           X_train=X_train, 
                           y_train=y_train, 
                           train_dataset=c['train_datasets'])
    
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
    for idx, prediction in enumerate(predicted_test):
        if prediction > 0.5:
            print(y_test[idx], prediction)
    y_test = y_test.cpu().detach().numpy()

    # plot metrics
    plot_acc_and_loss(c=c, 
                      train_acc_history=train_acc_history, 
                      train_loss_history=train_loss_history,
                      val_acc_history=val_acc_history, 
                      val_loss_history=val_loss_history, 
                      test_acc=test_acc, 
                      test_loss=test_loss, 
                      train_dataset=c['train_datasets'],
                      test_dataset=c['train_datasets'],
                      model_type=c['model'])

    # calculate AUROC and AUPRC and plot metrics
    fpr, tpr, roc_auc = auroc_roc_curve(y_test, predicted_test)
    plot_auroc_roc_curve(c=c, 
                         fpr=fpr, 
                         tpr=tpr, 
                         roc_auc=roc_auc, 
                         train_dataset=c['train_datasets'], 
                         test_dataset=c['train_datasets'],
                         model_type=c['model'])
    
    precision, recall, prc_auc = auprc_prc_curve(y_test, predicted_test)
    plot_auprc_prc_curve(c=c, 
                         precision=precision, 
                         recall=recall, 
                         prc_auc=prc_auc, 
                         train_dataset=c['train_datasets'], 
                         test_dataset=c['test_datasets'],
                         model_type=c['model'])
    
    cfm = get_confusion_matrix(y_test, predicted_test)
    plot_confusion_matrix(c=c, 
                          confusion_matrix=cfm, 
                          train_dataset=c['train_datasets'], 
                          test_dataset=c['train_datasets'],
                          model_type=c['model'], 
                          percentage=True)
    plot_confusion_matrix(c=c, 
                          confusion_matrix=cfm, 
                          train_dataset=c['train_datasets'], 
                          test_dataset=c['train_datasets'],
                          model_type=c['model'], 
                          percentage=False)
    
    # load test data
    positive_test_tensors = {}
    negative_test_tensors = {}
    for dataset in c['test_datasets']:
        positive_test_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt')))
        negative_test_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt')))
        
    LOOX_test, LOOy_test = get_tensors_to_test(positive_test_tensors, negative_test_tensors)
    LOO_test_set = BERTDataset(c, device, LOOX_test, LOOy_test)
    LOO_loader = get_LOO_dataset_loader(LOO_test_set, c['batch_size'])    
    
    # evaluate on test data
    LOO_predicted_test, LOO_test_acc, LOO_test_loss = trainer.predict(LOO_loader, cost)
    LOOy_test = LOOy_test.cpu().detach().numpy()
    
    # plot LOO metrics
    
    # calculate AUROC and AUPRC and plot metrics
    fpr, tpr, roc_auc = auroc_roc_curve(LOOy_test, LOO_predicted_test)
    plot_auroc_roc_curve(c=c, 
                         fpr=fpr, 
                         tpr=tpr, 
                         roc_auc=roc_auc, 
                         train_dataset=c['train_datasets'], 
                         test_dataset=c['test_datasets'], 
                         model_type=c['model'])
    
    precision, recall, prc_auc = auprc_prc_curve(LOOy_test, LOO_predicted_test)
    plot_auprc_prc_curve(c=c, 
                         precision=precision, 
                         recall=recall, 
                         prc_auc=prc_auc, 
                         train_dataset=c['train_datasets'], 
                         test_dataset=c['test_datasets'], 
                         model_type=c['model'])
    
    cfm = get_confusion_matrix(LOOy_test, LOO_predicted_test)
    plot_confusion_matrix(c=c, 
                          confusion_matrix=cfm, 
                          train_dataset=c['train_datasets'], 
                          test_dataset=c['test_datasets'], 
                          model_type=c['model'], 
                          percentage=True)
    plot_confusion_matrix(c=c, 
                          confusion_matrix=cfm, 
                          train_dataset=c['train_datasets'], 
                          test_dataset=c['test_datasets'], 
                          model_type=c['model'], 
                          percentage=False)
    
    plot_acc_and_loss(c=c, 
                      train_acc_history=train_acc_history, 
                      train_loss_history=train_loss_history,
                      val_acc_history=val_acc_history, 
                      val_loss_history=val_loss_history, 
                      test_acc=LOO_test_acc, 
                      test_loss=LOO_test_loss, 
                      train_dataset=c['train_datasets'],
                      test_dataset=c['test_datasets'],
                      model_type=c['model'])
    
    
    
