import math
import numpy as np
import random

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