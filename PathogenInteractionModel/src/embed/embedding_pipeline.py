import os, sys
import torch
from tqdm import tqdm

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_normalize import normalize

def embedding_pipeline_for_recapture(c, dataset, embedding_model, embedding_tokenizer, device):
    encoding_info = {}
    for new_ip in tqdm(os.listdir(os.path.join(c[f'{dataset}_dir'], 'new', 'text'))[0:10000]):
        with open(os.path.join(c[f'{dataset}_dir'], 'new', 'text', new_ip), 'r') as f:
            text = f.read()
        text = normalize(text)
        tokens = text.split(' ')
        split_tokens = [' '.join(tokens[i:i+c['max_seq_length']]) for i in range(0, len(tokens), c['max_seq_length'])]
    
        encoding = torch.zeros(1024).to(device)
        with torch.no_grad():
            for text_data in split_tokens:
                input_ids = torch.tensor(embedding_tokenizer.encode(text_data)).unsqueeze(0).to(device)
                try:
                    output = embedding_model(input_ids).last_hidden_state.squeeze(0)[0]/len(split_tokens)
                except:
                    output = embedding_model(input_ids)[-1].squeeze(0)/len(split_tokens)
                del input_ids
                encoding += output
            encoding_info[new_ip] = encoding
    torch.save(encoding_info, os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'new.pt'))  

def embedding_pipeline_for_new(c, dataset, embedding_model, embedding_tokenizer, device):
    encoding_info = {}
    for new_ip in tqdm(os.listdir(os.path.join(c[f'{dataset}_dir'], 'new'))):
        with open(os.path.join(c[f'{dataset}_dir'], 'new', new_ip), 'r') as f:
            text = f.read()
        text = normalize(text)
        tokens = text.split(' ')
        split_tokens = [' '.join(tokens[i:i+c['max_seq_length']]) for i in range(0, len(tokens), c['max_seq_length'])]
    
        encoding = torch.zeros(1024).to(device)
        with torch.no_grad():
            for text_data in split_tokens:
                input_ids = torch.tensor(embedding_tokenizer.encode(text_data)).unsqueeze(0).to(device)
                try:
                    output = embedding_model(input_ids).last_hidden_state.squeeze(0)[0]/len(split_tokens)
                except:
                    output = embedding_model(input_ids)[-1].squeeze(0)/len(split_tokens)
                del input_ids
                encoding += output
            encoding_info[new_ip] = encoding
    torch.save(encoding_info, os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'new.pt'))  

def embedding_pipeline(c, dataset, embedding_model, embedding_tokenizer, positives, negatives, device):
    positives_encoding_info = {}
    for positive in tqdm(positives):
        with open(os.path.join(c[f'{dataset}_dir'], 'text', positive), 'r') as f:
            text = f.read()
        text = normalize(text)
        tokens = text.split(' ')
        split_tokens = [' '.join(tokens[i:i+c['max_seq_length']]) for i in range(0, len(tokens), c['max_seq_length'])]
        
        encoding = torch.zeros(1024).to(device)
        with torch.no_grad():
            for text_data in split_tokens:
                input_ids = torch.tensor(embedding_tokenizer.encode(text_data)).unsqueeze(0).to(device)
                try: 
                    output = embedding_model(input_ids).last_hidden_state.squeeze(0)[0]/len(split_tokens)
                except:
                    output = embedding_model(input_ids)[-1].squeeze(0)/len(split_tokens)
                del input_ids
                encoding += output
            positives_encoding_info[positive] = encoding
        
    torch.save(positives_encoding_info, os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'positives.pt'))        
    
    
    negatives_encoding_info = {}      
    for negative in tqdm(negatives):
        with open(os.path.join(c[f'{dataset}_dir'], 'text', negative), 'r') as f:
            text = f.read()
        text = normalize(text)
        tokens = text.split(' ')
        split_tokens = [' '.join(tokens[i:i+c['max_seq_length']]) for i in range(0, len(tokens), c['max_seq_length'])]
        
        encoding = torch.zeros(1024).to(device)
        with torch.no_grad():
            for text_data in split_tokens:
                input_ids = torch.tensor(embedding_tokenizer.encode(text_data)).unsqueeze(0).to(device)
                try:
                    output = embedding_model(input_ids).last_hidden_state.squeeze(0)[0]/len(split_tokens)
                except:
                    output = embedding_model(input_ids)[-1].squeeze(0)/len(split_tokens)
                encoding += output
            negatives_encoding_info[negative] = encoding
            
    torch.save(negatives_encoding_info, os.path.join(c[f'{dataset}_dir'], 'model_embeddings', 'negatives.pt'))       
    return