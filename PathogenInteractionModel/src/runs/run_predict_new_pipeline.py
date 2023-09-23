import os, sys
import torch
import pandas as pd

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from embed.embedding_models import load_bert_model_from_pretrained
from embed.embedding_pipeline import embedding_pipeline_for_new
from training.embedding_models import LogisticRegression
from utils.metrics import plot_new_data_pca, plot_new_data_isomap
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
            embedding_pipeline_for_new(c, dataset_to_predict, model, tokenizer, device)
        
        # run the prediction pipeline
        # load the embeddings
        tensor_dicts = torch.load(os.path.join(c[f'{dataset_to_predict}_dir'], 'model_embeddings', 'new.pt'))
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
    
    # save the predictions
    df = pd.DataFrame({'filename': filenames, 'prediction': predicted_classification, 'rounded_prediction': predicted_classification.round()})
    df.to_csv(os.path.join(c[f'{dataset_to_predict}_dir'], f'{dataset_to_predict}_{c["model"]}_new_predicted_values.csv'), index=False)