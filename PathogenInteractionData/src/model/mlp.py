import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_operations import ngram_vectorize

def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()
    
def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 1),
                                          num_ngrams=50):
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    vectorized_texts = vectorizer.fit_transform(sample_texts)

    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))

    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()

def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model
# 1e-4, 2, 4, 0.2
def train_ngram_model(c, data,
                      learning_rate=1e-4, 
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.4,
                      ngram_range=(1,1)):
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = 2 #explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(2)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, name_train, vectorizer = ngram_vectorize(c,
        train_texts, train_labels, val_texts, ngram_range)
    
    x_train, x_test, name_test, vectorizer = ngram_vectorize(c ,
        train_texts, train_labels, test_texts, ngram_range)

    
    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'
    #optimizer = adam_v2.Adam(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    #callbacks = [tf.keras.callbacks.EarlyStopping(
        #monitor='val_loss', patience=2)]

    # Train and validate model.
    classifier = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            #callbacks=callbacks,
            #validation_split=0.2,
            validation_data=(x_val, val_labels),
            verbose=2, 
            batch_size=batch_size)

    # Print results.
    history = classifier.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    #model.save('VIP_mlp_model.h5')
    
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, test_labels, batch_size=batch_size)
    print('test loss, test acc:', results)
    
    return model, history, history['val_acc'][-1], history['val_loss'][-1], results[1], results[0], classifier