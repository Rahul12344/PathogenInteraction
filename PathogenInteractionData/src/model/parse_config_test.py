import numpy as np
import pandas as pd
import logging
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Grid search for tuning hyperparameters")
    parser.add_argument('-configFile', type = str, default = 'none', help = "Config file for running the code")
    parser.add_argument('-outFile', type = str, default = 'none', help = "Config file for running the code")

    arg_vals = parser.parse_args()
    inputFile=arg_vals.configFile
    outFile = arg_vals.outFile
    print(inputFile)