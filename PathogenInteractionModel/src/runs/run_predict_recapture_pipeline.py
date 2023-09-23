import os, sys
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from embed.embedding_models import load_bert_model_from_pretrained
from embed.embedding_pipeline import embedding_pipeline_for_recapture
from training.embedding_models import LogisticRegression
from utils.metrics import plot_new_data_pca, plot_new_data_isomap
from utils.train_test_split import train_test_split_keys, get_text_data
from training.train_model import train_ngram_model
from utils.train_test_split import get_tensors_to_test

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # load config from config.yaml
    c = config()
    
    dataset_to_predict = c['dataset_to_predict']
    
    if c['model'] != 'mlp':
        positive_tensors = {}
        negative_tensors = {}
        for dataset in c['train_datasets']:
            positive_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt')))
            negative_tensors.update(torch.load(os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt')))
        
        X, y = get_tensors_to_test(positive_tensors, negative_tensors)
        
        # run the embedding pipeline
        if c['should_generate_embeddings']:
            # load BERT model
            if c['embedding_model'] == 'bluebert':
                tokenizer, model = load_bert_model_from_pretrained(c, device)
            embedding_pipeline_for_recapture(c, dataset_to_predict, model, tokenizer, device)
        
        # run the prediction pipeline
        # load the embeddings
        tensor_dicts = torch.load(os.path.join(c[f'{dataset_to_predict}_dir'], 'model_embeddings', 'recapture.pt'))
        filenames = [k for k, _ in tensor_dicts.items()]
        tensors = [t for _, t in tensor_dicts.items()]
        tensors = torch.stack(tensors)
        
        plot_new_data_pca(c=c, 
                            X=X, 
                            y=y,
                            X_new=tensors,
                            dataset=c['dataset_to_predict'])
        
        plot_new_data_isomap(c=c,
                            X=X,
                            y=y,
                            X_new=tensors,
                            dataset=c['dataset_to_predict'])
        
        # load model state dict
        classifier = LogisticRegression(c=c, device=device)
        classifier.load_state_dict(torch.load(os.path.join(c['model_save_path'], f'{c["model"]}_{c["dataset_to_predict"]}_trained.pt')))
        classifier.eval()
        
        predicted_classification = classifier(tensors).squeeze().cpu().detach().numpy()
    
    else:
        NUM_TRIALS = 10 #10
        df_data = {}
        recapture_file_type = "negrecapture"
        df_test_data = {}
        
        recapture_filenames = os.listdir(os.path.join(c[f'{dataset_to_predict}_dir'], f'{recapture_file_type}', 'text'))
        df_data['Pubmed IDs'] = recapture_filenames
        for trial_num in tqdm(range(NUM_TRIALS)):
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
            
            texts = []
            for new_ip in tqdm(recapture_filenames):
                with open(os.path.join(c[f'{dataset_to_predict}_dir'], f'{recapture_file_type}', 'text', new_ip), 'r') as f:
                    text = f.read()
                    texts += [text]
                                
            transformed = vectorizer.transform(texts)
            transformed = selector.transform(transformed).astype('float32').todense()
            
            predicted_classification = model.predict(transformed)
            predicted_classification = np.array(predicted_classification)
            predicted_classification = predicted_classification.flatten()
            df_data[f'Trial {trial_num} Predicted'] = predicted_classification
            df_data[f'Trial {trial_num} Rounded'] = predicted_classification.round()
            
            df_test_data[f'Trial {trial_num} Predicted'] = np.array(predicted_test).flatten()
            df_test_data[f'Trial {trial_num} Rounded'] = np.array(predicted_test).flatten().round()
            df_test_data['True'] = np.array(y_test).flatten()
           
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(c[f'{dataset_to_predict}_dir'], f'{dataset_to_predict}_{c["model"]}_{recapture_file_type}_predicted_values_{NUM_TRIALS}_trials_relu.csv'), index=False)
        df = pd.DataFrame(df_test_data)
        df.to_csv(os.path.join(c[f'{dataset_to_predict}_dir'], f'{dataset_to_predict}_{c["model"]}_test_predicted_values_{NUM_TRIALS}_trials_relu.csv'), index=False)
        
    # save the predictions
    df = pd.DataFrame({'filename': filenames, 'prediction': predicted_classification, 'rounded_prediction': predicted_classification.round()})
    df.to_csv(os.path.join(c[f'{dataset_to_predict}_dir'], f'{dataset_to_predict}_{c["model"]}_{recapture_file_type}_predicted_values.csv'), index=False)