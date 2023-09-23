import sys, os
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import shap
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import statistics
from sklearn.metrics import f1_score, confusion_matrix
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_util import set_plot_info
from model.mlp import train_ngram_model
from utils.text_operations import ngram_vectorize
from data_loader.load_dataset import load_data, load_new_data
from data_loader.write_dataset import write_data
from config.config import config

NGRAM_RANGE = (1,1)

shap.initjs()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tf.autograph.set_verbosity(0)

if __name__ == '__main__':
    c = config()
    set_plot_info(c)
    
    logger.info('Logging results for number of epochs: {0}'.format(c['epochs']))
        
    box_plot_dict = defaultdict(list)   
    df = pd.DataFrame() 
    for iterations in range(0, c['trials']): 
        seed=123244
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(c, positive_percentage=0.3, seed=seed, dataset_size=1500, training_size=0.6,testing_size=0.2,validation_size=0.2)
        logger.info("Training on ", len(X_train), " validating on ", len(X_val))
        model, history, acc, loss, res_acc, res_loss, classifier = train_ngram_model(c, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), epochs=215, ngram_range=NGRAM_RANGE)

        x_train, x_test, names, vectorizer = ngram_vectorize(
                    c, X_train, y_train, X_test, ngram_range=NGRAM_RANGE)
        y_pred_keras = model.predict(x_test).ravel()

        X_train_ = np.array(x_train)
        tf.convert_to_tensor(X_train_.flatten())
        background = X_train_#[np.random.choice(X_train_.shape[0], 500, replace=False)]
        e = shap.DeepExplainer(model, background)
        
        X_test_ = np.array(x_test)
        X_labels = np.array(names)
        tf.convert_to_tensor(X_test_.flatten())
        tf.convert_to_tensor(X_labels.flatten())
        
        shap_values = e.shap_values(background)
        feature_order = np.flip(np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)))

        shap_val_sum = np.flip(np.sort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)))
        
        if c['box-plot']:
            for i in range(0,len(shap_val_sum)):
                if X_labels[feature_order[i]] not in box_plot_dict:
                    box_plot_dict[X_labels[feature_order[i]]] = []
                box_plot_dict[X_labels[feature_order[i]]].append(shap_val_sum[i])
            
        seed +=1
        if c['shap-summary']:
            shap.summary_plot(shap_values, background, X_labels)
        

        if c['confusion-matrix']:
            cf_matrix = confusion_matrix(y_test, y_pred_keras.round(), normalize=None)
            
            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in
                            cf_matrix.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in
                                cf_matrix.flatten()/np.sum(cf_matrix)]
            labels = ["{}\n{}\n{}".format(v1,v2,v3) for v1, v2, v3 in
                    zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)

            sns.heatmap(cf_matrix, annot=labels, fmt='')
                    
            plt.show()
            
            f1 = f1_score(y_test, y_pred_keras.round(), average='macro')
        
        
        if c['output-diff']:
            transformer = TfidfTransformer()
            loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vectorizer.vocabulary_)
            logger.info("vectorizer loaded")
            
            abstracts, data = load_new_data('/Users/rahulnatarajan/Research')
            logger.info("data loaded")
            df['Abstracts ID'] = abstracts
            
            testing = [1] * len(data)
            texts = transformer.fit_transform(loaded_vec.fit_transform(np.array(data)))
            texts = texts.astype('float32').todense()

            test_vals = np.squeeze(np.asarray(texts))
            predictions = model.predict(texts).ravel()
            predictions = np.round(predictions, 0)
            df = write_data(predictions, iterations)
            
        seed+=1
    
    if c['output-diff']: 
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        df.to_csv(os.path.join(c['output_dir'], 'comparison', f"{c['dataset']}_prediction_diff_{len(predictions[0])}_trials_{current_time_str}.tsv"), index=False, sep='\t') 
    
    data = []
    keys = box_plot_dict.keys()
    if c['box-plot']: 
        median_sorted = sorted(box_plot_dict.items(), key=lambda k: statistics.median(k[1]), reverse=True)
            
        keys = [item[0] for item in median_sorted][0:10]
        data = [item[1] for item in median_sorted][0:10]
            
        fig, ax = plt.subplots() 
        ax.set_xlabel('Feature Name')
        ax.set_ylabel('SHAP Value')
        ax.margins(x=0.005)
        
        positions = []
        curr = 0.3
        for i in range(len(data)):
            positions.append(curr)
            curr += 0.3
            
        bp = ax.boxplot(data, patch_artist=True, positions=positions, widths=[0.25]*len(data))
        for i in range(len(data)):
            x = np.random.normal(positions[i]-1, 0.04, size=len(data[i]))
            print(x)
            ax.scatter([a+1 for a in x], data[i], color='orange', zorder=100)
        plt.xticks(list(map(lambda x: x * 0.3, list(range(1,len(data)+1)))), keys)
        plt.xticks(rotation=90)
        
        i = 0
        for box in bp['boxes']:
            box.set(color='black', linewidth=2)
            box.set(facecolor = 'olive' )
            i+=1
        
        plt.show()