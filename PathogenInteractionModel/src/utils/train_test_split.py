from sklearn.model_selection import train_test_split
import torch
import os
import random
import nltk
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english'))

# split the data into train validation and test sets with a 60:20:20 split
def train_test_split_keys(c, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=c['random_state'], shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=c['random_state'], shuffle=True)
    return set(X_train), set(X_val), set(X_test), y_train, y_val, y_test

# get train, validation, test tensors from dictionary  
def get_tensors_to_train(X_train_keys, X_val_keys, X_test_keys, positive_tensors, negative_tensors):
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []
    for label, tensors in [(1,positive_tensors), (0,negative_tensors)]:
        for tensor_key in tensors:
            if tensor_key in X_train_keys:
                X_train.append(tensors[tensor_key])
                y_train.append(label)
            elif tensor_key in X_val_keys:
                X_val.append(tensors[tensor_key])
                y_val.append(label)
            elif tensor_key in X_test_keys:
                X_test.append(tensors[tensor_key])
                y_test.append(label)
                
    # shuffle the data
    temp = list(zip(X_train, y_train))
    random.shuffle(temp)
    X_train, y_train = zip(*temp)
    temp = list(zip(X_val, y_val))
    random.shuffle(temp)
    X_val, y_val = zip(*temp)
    temp = list(zip(X_test, y_test))
    random.shuffle(temp)
    X_test, y_test = zip(*temp)
    
    return torch.stack(X_train), torch.stack(X_val), torch.stack(X_test), torch.FloatTensor(y_train), torch.FloatTensor(y_val), torch.FloatTensor(y_test)

def get_positive_tensors(X_keys, positive_tensors):
    X = []
    y = []
    for label, tensors in [(1,positive_tensors)]:
        for tensor_key in tensors:
            if tensor_key in X_keys:
                X.append(tensors[tensor_key])
                y.append(label)
    return torch.stack(X), torch.FloatTensor(y)

def get_tensors_to_test(positive_tensors, negative_tensors):
    X_test = []
    y_test = []
    for label, tensors in [(1,positive_tensors), (0,negative_tensors)]:
        for tensor_key in tensors:
            X_test.append(tensors[tensor_key])
            y_test.append(label)
                
    # shuffle the data
    temp = list(zip(X_test, y_test))
    random.shuffle(temp)
    X_test, y_test = zip(*temp)
    
    return torch.stack(X_test), torch.FloatTensor(y_test)

def text_data(c, positives, negatives, positives_datasets, negatives_datasets):
    X = []
    y = []
    for label, filenames in [(1,zip(positives, positives_datasets)), (0,zip(negatives, negatives_datasets))]:
        for filename, dataset in filenames:
            X.append(read_filename(c, dataset, filename))
            y.append(label)
    return X, y
        

def get_text_data(c, X_train_keys, X_val_keys, X_test_keys, positives, negatives, positives_datasets, negatives_datasets):
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []
    for label, filenames in [(1,zip(positives, positives_datasets)), (0,zip(negatives, negatives_datasets))]:
        for filename, dataset in filenames:
            if filename in X_train_keys:
                if os.path.isfile(os.path.join(c[f'{dataset}_dir'], 'text', filename)):
                    X_train.append(read_filename(c, dataset, filename))
                    y_train.append(label)
                else:
                    print(label, "NOT FOUND", os.path.join(c[f'{dataset}_dir'], 'text', filename))
            elif filename in X_val_keys:
                if os.path.isfile(os.path.join(c[f'{dataset}_dir'], 'text', filename)):
                    X_val.append(read_filename(c, dataset, filename))
                    y_val.append(label)
                else:
                    print(label, "NOT FOUND", os.path.join(c[f'{dataset}_dir'], 'text', filename))
            elif filename in X_test_keys:
                if os.path.isfile(os.path.join(c[f'{dataset}_dir'], 'text', filename)):
                    X_test.append(read_filename(c, dataset, filename))
                    y_test.append(label)
                else:
                    print(label, "NOT FOUND", os.path.join(c[f'{dataset}_dir'], 'text', filename))
            
    # shuffle the data
    temp = list(zip(X_train, y_train))
    random.shuffle(temp)
    X_train, y_train = zip(*temp)
    temp = list(zip(X_val, y_val))
    random.shuffle(temp)
    X_val, y_val = zip(*temp)
    temp = list(zip(X_test, y_test))
    random.shuffle(temp)
    X_test, y_test = zip(*temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def read_filename(c, dataset, filename):
    with open(os.path.join(c[f'{dataset}_dir'], 'text', filename)) as f:
        filtered_sentence = []
        lower_cased_ = f.read().lower()
        lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
        for w in lower_cased.split(' '): 
            if w not in stop_words: 
                filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)