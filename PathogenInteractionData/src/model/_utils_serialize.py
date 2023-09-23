import pickle
import os
import random
import numpy as np
import argparse
import nltk
from nltk.corpus import stopwords 
import math

stop_words = set(stopwords.words('english'))

def serialize(dictionary, path):
    serialized = pickle.dumps(dictionary)
    file = open(path, "wb")
    file.seek(0)
    file.write(serialized)
    file.close()
    
def load_data(data_path="/Users/rahulnatarajan/Research/", positive_percentage=0.3, seed=123244, dataset_size=500, training_size=0.8, testing_size=0.2, validation_size=0.0, data="VIP"):
    training_pos_size = training_size * float(dataset_size) * positive_percentage
    training_neg_size = training_size * float(dataset_size) * (1-positive_percentage)
    
    testing_pos_size = testing_size * float(dataset_size) * positive_percentage
    testing_neg_size = testing_size * float(dataset_size) * (1-positive_percentage)
    
    validation_pos_size = validation_size * float(dataset_size) * positive_percentage
    validation_neg_size = validation_size * float(dataset_size) * (1-positive_percentage)
    
    path_to_data = os.path.join(data_path, 'Garud/train/OLD/')
    
    training_data = "training"
    validation_data = "validation"
    testing_data = "testing"
    
    if data == "malaria":
        training_data = "malaria_training"
        validation_data = "malaria_validation"
        testing_data = "malaria_testing"
    if data == "bacteria":
        training_data = "bacteria_train"
        validation_data = "bacteria_validation"
        testing_data = "bacteria_test"
    if data == "update_bacteria":
        training_data = "updated_bacteria_train"
        validation_data = "updated_bacteria_validation"
        testing_data = "updated_bacteria_test"
    
    #path_to_data = os.path.join(data_path, 'train')
    train_texts = []
    train_labels = []
    training_pos = len(os.listdir(os.path.join(path_to_data, training_data, "pos")))
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, training_data, category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    #lower_cased = f.read().lower()
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
    testing_pos = len(os.listdir(os.path.join(path_to_data, testing_data, "pos")))
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
    
def split_into_kfolds(input_x, input_y, k):
    size = len(input_x)
    k_fold_size = math.floor(size * float(1)/k)
    training_x_sets = []
    training_y_sets = []
    validation_x_sets = []
    validation_y_sets = []
    for i in range(k):
        num_in_k = 0
        training_x_sets.append(list())
        training_y_sets.append(list())
        validation_x_sets.append(list())
        validation_y_sets.append(list())
        while num_in_k < k_fold_size:
            position = random.randint(0, size-1)
            item_already_placed = False
            for j in range(0, i+1):
                if input_x[position] in validation_x_sets[j]:
                    item_already_placed = True
            if not item_already_placed:
                validation_x_sets[i].append(input_x[position])
                validation_y_sets[i].append(input_y[position])
                num_in_k += 1
        for index in range(size):
            if input_x[index] not in validation_x_sets[i]:
                training_x_sets[i].append(input_x[index])
                training_y_sets[i].append(input_y[index])
    
    training_y_sets_fixed = list()
    validation_y_sets_fixed = list()
    for train_y in training_y_sets:
        training_y_sets_fixed.append(np.array(train_y))
    for val_y in validation_y_sets:
        validation_y_sets_fixed.append(np.array(val_y))
    
    
    return training_x_sets, training_y_sets_fixed, validation_x_sets, validation_y_sets_fixed

if __name__ == "__main__":
    # read in command-line arguments
    parser = argparse.ArgumentParser(description= "Serialize given dataset")
    parser.add_argument('-dataset', type = str, default = 'VIP', help = "Dataset to serialize")
    parser.add_argument('-k', type = int, default = '5', help = "K-Fold size")

    arg_vals = parser.parse_args()
    dataset=arg_vals.dataset   
    k = arg_vals.k
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(data=dataset)     
    training_x_sets, training_y_sets_fixed, validation_x_sets, validation_y_sets_fixed = split_into_kfolds(train_x, train_y, k) 
    
    dictionary = {}
    dictionary['train_validation'] = (training_x_sets, training_y_sets_fixed, validation_x_sets, validation_y_sets_fixed)
    dictionary['test'] = (test_x, test_y)
                                     
    serialize(dictionary, "./data/vip_data")