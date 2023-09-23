# Setup

We use Anaconda to manage our packages. NOTE: SETUP SPECIFICALLY FOR MAC OS. In the root 
directory of the project, type `conda create --name [env_nam] --file spec-file.txt` and 
activate the environment by typing `conda activate [env_name]`. If any issues occur, check 
resolution through the Anaconda website. On other platforms, inspect the package list
denoted in spec-file.txt and download the platform-specific packages using conda, making 
sure to note the versions of the packages.

# Components

# DownloadWorker

Downloads PubMed abstracts corresponding to a range of PubMed IDs for future text analysis.
In the root directory, start the DownloadWorker by typing 
`python train/main.py --pos_size=[number_of_positives] --neg_size=[number_of_negatives] 
--dataset=[VIP/VIPextra/malaria/bacteria/bacteria_update/bacteria_prediction/bacteria_dl]
--path=[[root]/train/dataset/] --val_size=[0.0-1.0] --test_size=[0.0-1.0]`

We download by querying PubMed and download the corresponding abstracts using Python's 
Threading module.

# Hyperparameter tuning
We tune our Multi-layered perceptron model by testing/training/validating on 1500 VIP abstracts
over a set of 4480 possible configurations, where our parameters are learning rate, n-gram size,
dropout rate, units per layer, and number of layers.

In the root directory, we first need to establish our dataset to tune our hyperparameters on. First,
run `python train/_utils_serialize.py` with the default CLI arguments. Then, run 
`python train/grid_search_02122022.py`.

Note the optimal hyperparameters and use them for the model going forward.

# All utilities
Modify the main function in the file to call the necessary function. For example, if we want to generate 
box plots and predict the bacteria dataset, we can call `create_comparison` with bacteria as our dataset 
argument.

# Date Distribution
To analyze the distribution of dates of bacteria abstracts, run python `python train/date_grid_bacteria.py`