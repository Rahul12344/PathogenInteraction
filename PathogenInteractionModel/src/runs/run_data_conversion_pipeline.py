import os, sys
import pandas as pd
from tqdm import tqdm

# add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config

if __name__ == '__main__':
    # load config from config.yaml
    c = config()
    
    if c['should_create_frame']:
        df_data = []
        for dataset in ['bacteria', 'virus', 'malaria']:
            for data_class in ['training', 'validation', 'testing']:
                for label in ['pos', 'neg']:
                    for file_name in os.listdir(os.path.join(c['data-directory'], dataset+'_data', data_class, label)):
                        if '.txt' in file_name:
                            if label == 'pos':
                                df_data.append((file_name, 1, dataset))
                            elif label == 'neg':
                                df_data.append((file_name, 0, dataset))
        
        dataset_df = pd.DataFrame(df_data, columns=['file_name', 'label', 'dataset'])
        dataset_df.to_csv(os.path.join(c['data-directory'], 'dataset_df.csv'), index=False)
    
    for dataset in ['bacteria', 'virus', 'malaria']:
        for data_class in ['training', 'validation', 'testing']:
            for label in ['pos', 'neg']:
                if os.path.isdir(os.path.join(c['data-directory'], dataset+'_data', data_class, label)):
                    for file_name in os.listdir(os.path.join(c['data-directory'], dataset+'_data', data_class, label)):
                        if '.txt' in file_name:
                            path = os.path.join(c['data-directory'], dataset+'_data', data_class, label, file_name)
                            os.rename(path,  os.path.join(c['data-directory'], dataset+'_data', file_name))
                        
    for dataset in ['bacteria', 'virus', 'malaria']:
            for file_name in os.listdir(os.path.join(c['data-directory'], dataset+'_data')):
                if '.txt' in file_name:
                    path = os.path.join(c['data-directory'], dataset+'_data', file_name)
                    os.rename(path, os.path.join(c['data-directory'], dataset+'_data', 'text', file_name))