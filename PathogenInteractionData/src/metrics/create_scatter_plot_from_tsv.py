import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os

def create_scatter_plot(c, metric='Average AUC'):
    df = pd.read_csv(os.path.join(c['output_dir'], 'grid_search', f"{c['dataset']}_grid_search.tsv"), index=False, sep='\t') 
    optimal_auc_entry = df.loc[df['Average AUC'].idxmax()]
    optimal_f1_entry = df.loc[df['Average F1 Score'].idxmax()]
    optimal_f1_and_auc_entry = df.loc[df['Average F1 Score'] + df['Average AUC'].idxmax()]
    
    max_df = pd.DataFrame([optimal_auc_entry, optimal_f1_entry, optimal_f1_and_auc_entry], columns=df.columns)
    max_df.to_csv(os.path.join(c['output_dir'], 'grid_search', f"{c['dataset']}_optimal_f1_auc_hyperparameters.tsv"), index=False, sep='\t') 
    
    x_not_normalized = list(range(len(df)))
    x = [0.1*(a+1) for a in x_not_normalized]
    
    scatter_plot_learning_rate_layers(x, df[metric].tolist(), df, show=False)
    scatter_plot_dropout_rate_units(x, df[metric].tolist(), df, show=False)
        

def scatter_plot_learning_rate_layers(c, x, Y, df, show=True):
    _, ax = plt.subplots() 
    ax.set_xlabel('N-Gram Size', labelpad=10)
    ax.set_ylabel('AUC Value', labelpad=40)  
    
    legend_elements = [Line2D([0], [0], marker='v', color='w', label='Learning Rate: 1e-06, Layers: 1',
                          markerfacecolor='orange', markersize=7),
                        Line2D([0], [0], marker='v', color='w', label='Learning Rate: 1e-04, Layers: 1',
                          markerfacecolor='red', markersize=7),
                        Line2D([0], [0], marker='v', color='w', label='Learning Rate: 1e-02, Layers: 1',
                          markerfacecolor='blue', markersize=7),
                        Line2D([0], [0], marker='^', color='w', label='Learning Rate: 1e-06, Layers: 2',
                          markerfacecolor='orange', markersize=7),
                        Line2D([0], [0], marker='^', color='w', label='Learning Rate: 1e-04, Layers: 2',
                          markerfacecolor='red', markersize=7),
                        Line2D([0], [0], marker='^', color='w', label='Learning Rate: 1e-02, Layers: 2',
                          markerfacecolor='blue', markersize=7)]
    
    learning_rates_colors = {1e-06: "orange",
                              1e-04: "red", 
                              1e-02: "blue"}
    layers_markers = {1: "v", 
              2: "^"}
    
    learning_rates = df['Learning Rate'].tolist()
    num_layers = df['Num Layers'].tolist()
    
    for i, learning_rate, layer in enumerate(zip(learning_rates, num_layers)):
        ax.scatter(x[i], Y[i], color=learning_rates_colors[learning_rate], marker=layers_markers[layer], zorder=100)
    if show:
        ax.legend(handles=legend_elements, loc='best')
        x = [1, 2, 3, 4]
        xi = list([1.0, 5.0, 9.0, 13.0])
        plt.xticks(xi, x)
        plt.savefig(os.path.join(c['plot_dir'], 'grid_search', f"{c['dataset']}_learning_rate_layers_scatter_plot.png"))
        
def scatter_plot_dropout_rate_units(c, x, Y, df, show=True):
    fig, ax = plt.subplots() 
    ax.set_xlabel('N-Gram Size', labelpad=10)
    ax.set_ylabel('AUC Value', labelpad=40)   
    
    legend_elements = [Line2D([0], [0], marker='v', color='w', label='Dropout Rate: 0.2, Units/Layer: 8',
                          markerfacecolor='orange', markersize=7),
                        Line2D([0], [0], marker='v', color='w', label='Dropout Rate: 0.2, Units/Layer: 64',
                          markerfacecolor='red', markersize=7),
                        Line2D([0], [0], marker='v', color='w', label='Dropout Rate: 0.2, Units/Layer: 100',
                          markerfacecolor='blue', markersize=7),
                        Line2D([0], [0], marker='^', color='w', label='Dropout Rate: 0.4, Units/Layer: 8',
                          markerfacecolor='orange', markersize=7),
                        Line2D([0], [0], marker='^', color='w', label='Dropout Rate: 0.4, Units/Layer: 64',
                          markerfacecolor='red', markersize=7),
                        Line2D([0], [0], marker='^', color='w', label='Dropout Rate: 0.4, Units/Layer: 100',
                          markerfacecolor='blue', markersize=7)]
    
    dropout_rates_markers = {0.2: "v",
                              0.4: "^"}
    units_colors = {8: "orange", 
              64: "red", 
              100: "blue"}
    
    dropout_rates = df['Dropout Rate'].tolist()
    units = df['Units per Layer'].tolist()
    
    for i, dropout_rate, unit in enumerate(zip(dropout_rates, units)):            
        ax.scatter(x[i], Y[i], color=units_colors[unit], marker=dropout_rates_markers[dropout_rate], zorder=100)
    
    if show:
        ax.legend(handles=legend_elements, loc='best')
        x = [1, 2, 3, 4]
        xi = list([1.0, 5.0, 9.0, 13.0])
        plt.xticks(xi, x)
        plt.savefig(os.path.join(c['plot_dir'], 'grid_search', f"{c['dataset']}_dropout_rate_units_scatter_plot.png"))

def find_average_metric_under_param_value(df, parameter='Dropout Rate', value=0.2, metric="Average AUC"):
    return df[df[parameter] == value][metric].mean()     
        
    
