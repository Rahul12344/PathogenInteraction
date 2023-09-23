import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
now = datetime.now()

# calculate the area under the ROC curve
def auroc_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# calculate the area under the precision-recall curve
def auprc_prc_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prc_auc = auc(recall, precision)
    return precision, recall, prc_auc

# calculate the confusion matrix
def get_confusion_matrix(y_true, y_pred, threshold=0.5):
    return confusion_matrix(y_true, y_pred >= threshold)

# plot the ROC curve using seaborn
def plot_auroc_roc_curve(c, fpr, tpr, roc_auc, train_dataset, test_dataset, model_type):
    sns.set_style("darkgrid")
    roc_data = {"False Positive Rate": fpr, "True Positive Rate": tpr}
    roc_df = pd.DataFrame(roc_data)

    # Plot the AUROC curve using Seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=roc_df, x="False Positive Rate", y="True Positive Rate")
    plt.plot([0, 1], [0, 1], 'k--')  # Add a dashed diagonal line for reference
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve\nAUROC = {:.2f}'.format(roc_auc))
    plt.grid(True)

    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_{model_type}_roc_curve_{now}.png'))
    plt.clf()
    plt.close()
    
# plot the precision-recall curve using seaborn
def plot_auprc_prc_curve(c, precision, recall, prc_auc, train_dataset, test_dataset, model_type):
    sns.set_style("darkgrid")
    prc_data = {"Precision": precision, "Recall": recall}
    prc_df = pd.DataFrame(prc_data)

    # Plot the AUROC curve using Seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=prc_df, x="Recall", y="Precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve\nAUPRC = {:.2f}'.format(prc_auc))
    plt.grid(True)
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_{model_type}_prc_curve_{now}.png'))
    plt.clf()
    plt.close()
    
# plot the confusion matrix using seaborn
def plot_confusion_matrix(c, confusion_matrix, train_dataset, test_dataset, model_type, percentage=False):
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    if percentage:
        sns.heatmap(confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if percentage:
        plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_{model_type}_confusion_matrix_percentage_{now}.png'))
        plt.clf()
        plt.close()
    else:
        plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_{model_type}_confusion_matrix_raw_number_{now}.png'))
        plt.clf()
        plt.close()

# plot the training data in 2D using PCA using seaborn
def plot_training_data_pca(c, X_train, y_train, train_dataset):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train.cpu().detach().numpy())
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
    X_train_pca_df['label'] = y_train.cpu().detach().numpy()
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=X_train_pca_df, x='PC1', y='PC2', hue='label', palette='deep')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Training Data PCA')
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_{"_".join(train_dataset)}_training_data_pca_{now}.png'))
    plt.clf()
    plt.close()
    
# plot the training data in 2D using PCA using seaborn
def plot_new_data_pca(c, X, y, X_new, dataset):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.cpu().detach().numpy())
    X_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    X_df['label'] = y.cpu().detach().numpy()
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=X_df, x='PC1', y='PC2', hue='label', palette='deep')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    X_new_pca = pca.transform(X_new.cpu().detach().numpy())
    X_new_df = pd.DataFrame(X_new_pca, columns=['PC1', 'PC2'])
    sns.scatterplot(data=X_new_df, x='PC1', y='PC2', color='green', marker='x', label='New Data')
    plt.title('New Data PCA')
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_{dataset}_data_pca_{now}.png'))
    plt.clf()
    plt.close()
    
def plot_new_data_isomap(c, X, y, X_new, dataset):
    embedding = Isomap(n_components=2, n_neighbors=10)
    X_iso = embedding.fit_transform(X.cpu().detach().numpy())
    X_df = pd.DataFrame(X_iso, columns=['D1', 'D2'])
    X_df['label'] = y.cpu().detach().numpy()
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=X_df, x='D1', y='D2', hue='label', palette='deep')
    plt.xlabel('D1')
    plt.ylabel('D2')
    X_new_iso = embedding.transform(X_new.cpu().detach().numpy())
    X_new_df = pd.DataFrame(X_new_iso, columns=['D1', 'D2'])
    sns.scatterplot(data=X_new_df, x='D1', y='D2', color='green', marker='x', label='New Data')
    plt.title('New Data Isomap')
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_{dataset}_data_isomap_{now}.png'))
    plt.clf()
    plt.close()

def plot_acc_and_loss(c, train_acc_history, train_loss_history, val_acc_history, val_loss_history, test_acc, test_loss, train_dataset, test_dataset, model_type):
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    
    df = pd.DataFrame({'train_acc': train_acc_history, 'train_loss': train_loss_history, 'val_acc': val_acc_history, 'val_loss': val_loss_history, 'test_acc': [test_acc]*len(train_loss_history), 'test_loss': [test_loss]*len(train_loss_history), 'epoch': list(range(len(train_loss_history)))})
    sns.lineplot(data=df, x="epoch", y="train_acc", label='Train Accuracy')
    sns.lineplot(data=df, x="epoch", y="val_acc", label='Validation Accuracy')
    sns.lineplot(data=df, x="epoch", y="test_acc", label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_{model_type}_acc_history_{now}.png'))
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="epoch", y="train_loss", label='Train Loss')
    sns.lineplot(data=df, x="epoch", y="val_loss", label='Validation Loss')
    sns.lineplot(data=df, x="epoch", y="test_loss", label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["embedding_model"]}_train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_{model_type}_loss_history_{now}.png'))
    plt.clf()
    plt.close()
    
# plot the ROC curve using seaborn
def plot_auroc_roc_curve_comp(c, train_dataset, test_dataset, fpr_blue, tpr_blue, roc_auc_blue, fpr_mlp, tpr_mlp, roc_auc_mlp):
    sns.set_style("darkgrid")
    roc_data_blue = {"False Positive Rate": fpr_blue, "True Positive Rate": tpr_blue}
    roc_df_blue = pd.DataFrame(roc_data_blue)
    
    roc_data_mlp = {"False Positive Rate": fpr_mlp, "True Positive Rate": tpr_mlp}
    roc_df_mlp = pd.DataFrame(roc_data_mlp)

    # Plot the AUROC curve using Seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=roc_df_blue, x="False Positive Rate", y="True Positive Rate", label='BlueBERT AUROC = {:.2f}'.format(roc_auc_blue))
    sns.lineplot(data=roc_df_mlp, x="False Positive Rate", y="True Positive Rate", label='N-gram AUROC = {:.2f}'.format(roc_auc_mlp))
    plt.plot([0, 1], [0, 1], 'k--')  # Add a dashed diagonal line for reference
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for BlueBERT and MLP')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(c['metrics_dir'], f'train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_comparative_roc_curve_{now}.png'))
    plt.clf()
    plt.close()
    
# plot the precision-recall curve using seaborn
def plot_auprc_prc_curve_comp(c, train_dataset, test_dataset, precision_blue, recall_blue, prc_auc_blue, precision_mlp, recall_mlp, prc_auc_mlp):
    sns.set_style("darkgrid")
    prc_data_blue = {"Precision": precision_blue, "Recall": recall_blue}
    prc_df_blue = pd.DataFrame(prc_data_blue)
    
    prc_data_mlp = {"Precision": precision_mlp, "Recall": recall_mlp}
    prc_df_mlp = pd.DataFrame(prc_data_mlp)

    # Plot the AUROC curve using Seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=prc_df_blue, x="Recall", y="Precision", label='BlueBERT AUPRC = {:.2f}'.format(prc_auc_blue))
    sns.lineplot(data=prc_df_mlp, x="Recall", y="Precision", label='N-gram AUPRC = {:.2f}'.format(prc_auc_mlp))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve\nAUPRC = {:.2f}'.format(prc_auc_blue))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(c['metrics_dir'], f'train_{"_".join(train_dataset)}_test_{"_".join(test_dataset)}_comparative_prc_curve_{now}.png'))
    plt.clf()
    plt.close()
    
def create_dot_plot_of_transfer_learning_performance(c, df):
    """
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x='dataset', y='same_set_acc', label='Train Set Accuracy')
    sns.scatterplot(data=df, x='dataset', y='test_acc', alpha=0.5, label='Test Set Accuracy')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.title('Accuracy Comparison Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["model"]}_acc_comp.png'))
    plt.clf()
    plt.close()
    
    
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x='dataset', y='same_set_loss', label='Train Set Loss')
    sns.scatterplot(data=df, x='dataset', y='test_loss', label='Test Set Loss')
    plt.xlabel('Dataset')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.title('Loss Comparison Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["model"]}_loss_comp.png'))
    plt.clf()
    plt.close()
    """
    
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x='dataset', y='same_set_roc_auc', label='Train Set AUROC')
    sns.scatterplot(data=df, x='dataset', y='test_roc_auc', label='Test Set AUROC')
    plt.xlabel('Dataset')
    plt.ylabel('AUROC')
    plt.xticks(rotation=45)
    plt.title('AUROC Comparison Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["model"]}_auroc_comp.png'))
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x='dataset', y='same_set_prc_auc', label='Train Set AUPRC')
    sns.scatterplot(data=df, x='dataset', y='test_prc_auc', label='Test Set AUPRC')
    plt.xlabel('Dataset')
    plt.ylabel('AUPRC')
    plt.xticks(rotation=45)
    plt.title('AUPRC Comparison Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(c['metrics_dir'], f'{c["model"]}_auprc_comp.png'))
    plt.clf()
    plt.close()
    