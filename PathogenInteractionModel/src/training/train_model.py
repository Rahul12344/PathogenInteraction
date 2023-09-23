import os, sys
import torch
from torch.optim import Adam
from tensorflow.keras import optimizers
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import logging
import numpy as np
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

logging.basicConfig(level = logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_operations import ngram_vectorize
from training.mlp import mlp_model
"""
Class for training torch models
"""
class TrainModel:
    
    # initialize the class
    def __init__(self, c, classifier, epochs=110, lr=0.005, weight_decay=1e-5, min_delta=0.01, tolerance=5):
        self.c = c
        self.classifier = classifier
        self.epochs = epochs
        self.optimizer = Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        self.early_stopping_counter = 0
        self.min_delta = min_delta
        self.tolerance = tolerance
    
    # function to calculate the accuracy of the model at threshold of 0.5
    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(y_pred)
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
    
        return acc
    
    def early_stopping(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.early_stopping_counter +=1
            if self.early_stopping_counter >= self.tolerance:
                return True
        else:
            self.early_stopping_counter = 0
        return False
      
    # function to train the model  
    def fit(self, loss_function, train_dataset, val_dataset):
        train_acc_history, train_loss_history = [], []
        val_acc_history, val_loss_history = [], []
        for epoch in tqdm(range(self.epochs)):
            self.classifier.train()
            running_train_loss = 0.0
            running_train_acc = 0.0
            num_training_batches = 0 
            for batch in tqdm(train_dataset):
                num_training_batches += 1
                embeddings, true_classification = batch
                
                self.optimizer.zero_grad()
                predicted_classification = self.classifier(embeddings).squeeze()
                
                loss = loss_function(predicted_classification, true_classification, self.classifier.parameters())
                loss.backward()
                running_train_loss += loss.item()
                running_train_acc += self.binary_acc(predicted_classification, true_classification)
                self.optimizer.step()
            
            self.classifier.eval()
            running_val_loss = 0.0
            running_val_acc = 0.0
            num_val_batches = 0
            for batch in tqdm(val_dataset):
                num_val_batches += 1
                embeddings, true_classification = batch
                
                predicted_classification = self.classifier(embeddings).squeeze()
                
                loss = loss_function(predicted_classification, true_classification, self.classifier.parameters())
                running_val_loss += loss.item()
                running_val_acc += self.binary_acc(predicted_classification, true_classification)
            
            train_acc_history.append(running_train_acc.item()/num_training_batches)
            train_loss_history.append(running_train_loss/num_training_batches)
            val_acc_history.append(running_val_acc.item()/num_val_batches)
            val_loss_history.append(running_val_loss/num_val_batches)
            logging.info("Epoch {} - Training Loss: {} - Training Acc: {} - Validation Loss: {} - Validation Acc: {}". format(epoch+1, running_train_loss/num_training_batches, running_train_acc/num_training_batches, running_val_loss/num_val_batches, running_val_acc/num_val_batches))
        
            if (epoch+1) % 50 == 0 and self.c['save_checkpoint']:
                model_name = f"{self.c['model']}_{'_'.join(self.c['train_datasets'])}_epoch_{epoch+1}.pt"
                torch.save(self.classifier.state_dict(), os.path.join(self.c['model_save_path'], model_name))
        
            if self.early_stopping(running_train_loss/num_training_batches, running_val_loss/num_val_batches) and self.c['early_stopping']:
                break                                                                                             
                
        
        if self.c['save_model']:
            model_name = f"{self.c['model']}_{'_'.join(self.c['train_datasets'])}_trained.pt"
            torch.save(self.classifier.state_dict(), os.path.join(self.c['model_save_path'], model_name))
        return train_acc_history, train_loss_history, val_acc_history, val_loss_history
            
    def predict(self, test_dataset, loss_function):
        self.classifier.eval()
        running_test_acc = 0.0
        running_test_loss = 0.0
        num_test_batches = 0
        overall_predictions = []
        for batch in tqdm(test_dataset):
            num_test_batches += 1
            embeddings, true_classification = batch
            
            predicted_classification = self.classifier(embeddings).squeeze()
            overall_predictions.append(predicted_classification)
            
            running_test_acc += self.binary_acc(predicted_classification, true_classification)
            running_test_loss += loss_function(predicted_classification, true_classification, self.classifier.parameters()).item()
            
        overall_predictions = torch.cat(overall_predictions)   
        logging.info("Test Acc: {} Test Loss: {}".format(running_test_acc/num_test_batches, running_test_loss/num_test_batches))
        
        return overall_predictions.cpu().detach().numpy(), running_test_acc.item()/num_test_batches, running_test_loss/num_test_batches
    
"""
Function to train a ngram model
"""
def train_ngram_model(c, data,
                      learning_rate=1e-4, 
                      epochs=215,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.4,
                      ngram_range=(1,1),
                      drop_remainder=True):
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = data
    
    # convert all inputs to np array
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    val_texts = np.array(val_texts)
    val_labels = np.array(val_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)
    
    
    # Verify that validation labels are in the same range as training labels.
    num_classes = 2
    unexpected_labels = [v for v in val_labels if v not in range(2)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, name_train, vectorizer, selector = ngram_vectorize(c,
        train_texts, train_labels, val_texts, ngram_range)
    
    x_train, x_test, name_test, vectorizer, selector = ngram_vectorize(c ,
        train_texts, train_labels, test_texts, ngram_range)

    
    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  learning_rate=learning_rate,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # Train and validate model.
    classifier = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            #validation_split=0.2,
            validation_data=(x_val, val_labels),
            verbose=2, 
            batch_size=batch_size)

    # Print results.
    history = classifier.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    if c['save_model']:
        model_name = f"{c['model']}_{'_'.join(c['train_datasets'])}_trained.h5"
        model.save(os.path.join(c['model_save_path'], model_name))
    
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, test_labels, batch_size=batch_size)
    print('test loss, test acc:', results)
    
    predicted_test = model.predict(x_test)
    
    return model, predicted_test, vectorizer, selector

def cross_validate_torch_model():
    config = {
        "learning_rate": tune.grid_search([0.001, 0.01, 0.1]),  # Learning rate
        "epochs": tune.grid_search([10, 50, 100, 150]),  # Number of training epochs
    }
    
    analysis = tune.run(
        TrainModel,
        config=config,
        num_samples=1,  # Set the number of trials
        resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust resources as per your setup
        scheduler=ASHAScheduler(metric="accuracy", mode="max"),  # Set the scheduler and metric
        progress_reporter=CLIReporter(),
    )
    
    best_config = analysis.get_best_config(metric="accuracy", mode="max")
    best_accuracy = analysis.best_result["accuracy"]
    return best_config, best_accuracy

def cross_validate_keras_model(c,
                   data,
                   learning_rate=1e-4,
                   batch_size=128,
                   layers=2,
                   units=64,
                   epochs=215,
                   dropout_rate=0.4,
                   ngram_range=(1,1)):
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = data
    
    # convert all inputs to np array
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    val_texts = np.array(val_texts)
    val_labels = np.array(val_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)
    
    
    # Verify that validation labels are in the same range as training labels.
    num_classes = 2
    unexpected_labels = [v for v in val_labels if v not in range(2)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_test, name_train, vectorizer, selector = ngram_vectorize(c,
        train_texts, train_labels, test_texts, ngram_range)

    
    # Create model instance.
    model = KerasClassifier(model=mlp_model, verbose=0)

    parameters = {"layers":[2, 4, 8, 16],
                  "units":[2, 4, 16, 64, 256],
                  "dropout_rate":[0.1, 0.2, 0.3, 0.4, 0.5],
                  "input_shape":[x_train.shape[1:]],}
    
    grid = GridSearchCV(estimator=model, 
                        param_grid=parameters, 
                        cv=5, 
                        scoring='accuracy',
                        n_jobs=-1)
    
    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # Train and validate model.
    grid_result = grid.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2, 
            batch_size=batch_size)
    
    estimator = grid_result.best_estimator_
    
    print('\n# Evaluate on test data')
    results = estimator.evaluate(x_test, test_labels, batch_size=batch_size)
    print('test loss, test acc:', results)
    
    predicted_test = model.predict(x_test)
    return estimator, predicted_test