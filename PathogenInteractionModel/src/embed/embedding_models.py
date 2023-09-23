import os, sys
from transformers import BertTokenizer
from transformers import BertModel

def load_bert_model_from_pretrained(c, device):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(os.getcwd(), c['pretrained_model']))
    model = BertModel.from_pretrained(os.path.join(os.getcwd(), c['pretrained_model'])).to(device)
    return tokenizer, model