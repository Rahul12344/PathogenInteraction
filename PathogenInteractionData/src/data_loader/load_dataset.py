import nltk
from nltk.corpus import stopwords 
import os
import random, math
import numpy as np

stop_words = set(stopwords.words('english'))

def load_data(c, positive_percentage=0.3, seed=123244, dataset_size=450, training_size=0.6, testing_size=0.2, validation_size=0.2):
    training_pos_size = training_size * float(dataset_size) * positive_percentage
    training_neg_size = training_size * float(dataset_size) * (1-positive_percentage)
    
    testing_pos_size = testing_size * float(dataset_size) * positive_percentage
    testing_neg_size = testing_size * float(dataset_size) * (1-positive_percentage)
    
    validation_pos_size = validation_size * float(dataset_size) * positive_percentage
    validation_neg_size = validation_size * float(dataset_size) * (1-positive_percentage)
    
    path_to_data = c['model_data_dir']
    
    training_data = "training"
    validation_data = "validation"
    testing_data = "testing"
    
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, training_data, category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased_ = f.read().lower()
                    lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    train_texts.append(' '.join(filtered_sentence))
                train_labels.append(0 if category == 'neg' else 1)
                
    val_texts = []
    val_labels = []
    validation_pos = len(os.listdir(os.path.join(path_to_data, validation_data, "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, validation_data, category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    filtered_sentence = []
                    #lower_cased = f.read().lower()
                    lower_cased_ = f.read().lower()
                    lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    val_texts.append(' '.join(filtered_sentence))
                val_labels.append(0 if category == 'neg' else 1)
                
    test_texts = []
    test_labels = []
    test_abstracts = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, testing_data, category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased_ = f.read().lower()
                    lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    test_texts.append(' '.join(filtered_sentence))
                test_labels.append(0 if category == 'neg' else 1)
                test_abstracts.append(fname)
                
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)
    
    num_pos = 0
    num_neg = 0 
    train_texts_ = []
    train_labels_ = []
    val_texts_ = []
    val_labels_ = []
    test_texts_ = []
    test_labels_ = []
    test_abstracts_ = []
    
    for i in range(len(train_labels)):
        if train_labels[i] == 1 and num_pos < math.floor(training_pos_size):
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
            num_pos += 1
        elif train_labels[i] == 0 and num_neg < math.ceil(training_neg_size):
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
            num_neg += 1
    
    num_pos = 0
    num_neg = 0 
    for i in range(len(val_labels)):
        if val_labels[i] == 1 and num_pos < math.floor(validation_pos_size):
            val_texts_.append(val_texts[i])
            val_labels_.append(val_labels[i])
            num_pos += 1
        elif val_labels[i] == 0 and num_neg < math.ceil(validation_neg_size):
            val_texts_.append(val_texts[i])
            val_labels_.append(val_labels[i])
            num_neg += 1
    
    num_pos = 0
    num_neg = 0    
    for i in range(len(test_labels)):
        if test_labels[i] == 1 and num_pos < math.floor(testing_pos_size):
            test_texts_.append(test_texts[i])
            test_labels_.append(test_labels[i])
            test_abstracts_.append(test_abstracts[i])
            num_pos += 1
        elif test_labels[i] == 0 and num_neg < math.ceil(testing_neg_size):
            test_texts_.append(test_texts[i])
            test_labels_.append(test_labels[i])
            num_neg += 1

    return ((train_texts_, np.array(train_labels_)),
            (val_texts_, np.array(val_labels_)),
            (test_texts_, np.array(test_labels_)))
    
def load_new_data(c):
    texts = []
    abstracts = []
    path = os.path.join(c['model_data_dir'], 'new')
    for fname in sorted(os.listdir(path)):
        if fname.endswith('.txt'):
            with open(os.path.join(path, fname)) as f:
                filtered_sentence = []
                lower_cased_ = f.read().lower()
                lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                for w in lower_cased.split(' '): 
                    if w not in stop_words: 
                        filtered_sentence.append(w) 
                texts.append(' '.join(filtered_sentence))
                abstracts.append(fname)
    
    abstracts_ = []
    texts_ = []
    for i in range(len(texts)):
        abstracts_.append(abstracts[i])
        texts_.append(texts[i])
    return abstracts_, texts_