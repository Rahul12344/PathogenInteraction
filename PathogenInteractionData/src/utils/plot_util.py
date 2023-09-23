import matplotlib.pyplot as plt

def set_plot_info(c):
        plt.rcParams["font.family"] = c['font-family']

        plt.rc('font', size=c['font-size'])          # controls default text sizes
        plt.rc('axes', titlesize=c['font-size'])     # fontsize of the axes title
        plt.rc('axes', labelsize=c['font-size'])    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=c['font-size'])    # fontsize of the tick labels
        plt.rc('ytick', labelsize=c['font-size'])    # fontsize of the tick labels
        plt.rc('legend', fontsize=c['font-size'])    # legend fontsize
        plt.rc('figure', titlesize=c['font-size'])  # fontsize of the figure title