import os, sys
import logging
from sklearn.metrics import f1_score, roc_curve, auc
import pandas as pd
import datetime
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.load_dataset import load_data
from config.config import config
from utils.fold import split_into_kfolds
from model.mlp import train_ngram_model
from utils.text_operations import ngram_vectorize
from metrics.create_scatter_plot_from_tsv import create_scatter_plot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tf.autograph.set_verbosity(0)

if __name__ == '__main__':
    c = config()
    seed=123244
    
    logger.info('Logging results for number of epochs: {0}'.format(215))
        
    num_folds = 6
    
    (X_train_, y_train_), (X_val_, y_val_), (X_test, y_test) = load_data(c)
    X_train_list, y_train_list_, X_val_list, y_val_list = split_into_kfolds(X_train_, y_train_, num_folds)  
    
    ngram_sizes = [1,2,3,4]
    learning_rates = [1e-6, 1e-4, 1e-2]
    
    layers = [1,2]
    dropout_rates = [0.2, 0.4]
  
    units_per_layer = [8, 64, 100]
    
    itera = 0
    df_data = []
    for ngram_size in ngram_sizes: 
        for learning_rate in learning_rates:
            for layer in layers:
                for dropout_rate in dropout_rates:
                    for unit in units_per_layer:
                        total_auc = 0.0
                        total_f1 = 0.0
            
                        for (X_train, y_train, X_val, y_val) in zip(X_train_list, y_train_list_, X_val_list, y_val_list):
                            curr_n_gram = (ngram_size, ngram_size)
                            logger.info("training on ", len(X_train), " samples, validating on ", len(X_val))

                            model, history, acc, loss, r_acc, r_loss, classifier = train_ngram_model(((X_train, y_train), (X_val, y_val), (X_test, y_test)), epochs=215, ngram_range=curr_n_gram, learning_rate=learning_rate, layers=layer, units=unit, dropout_rate=dropout_rate)
                            x_train, x_test, names, vectorizer = ngram_vectorize(
                                    X_train, y_train, X_test, ngram_range=curr_n_gram)
                            y_pred_keras = model.predict(x_test).ravel()
                            
                            
                            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
                            auc_keras = auc(fpr_keras, tpr_keras)
                            total_auc += auc_keras
                            f1 = f1_score(y_test, y_pred_keras.round(), average='macro')
                            total_f1 += f1
                            
                        auc_ave = total_auc/6.0
                        f1_ave = total_f1/6.0
                        df_data.append((ngram_size, learning_rate, layer, dropout_rate, unit, auc_ave, f1_ave))
                        
    df = pd.DataFrame(df_data, columns=['Ngram Size', 'Learning Rate', 'Num Layers', 'Dropout Rate', 'Units per Layer', 'Average AUC', 'Average F1 Score'])
    df.to_csv(os.path.join(c['output_dir'], 'grid_search', f"{c['dataset']}_grid_search.tsv"), index=False, sep='\t') 
    
    if c['grid-search-plot']:
        create_scatter_plot(c, metric='Average AUC')