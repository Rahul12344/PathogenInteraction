import pandas as pd
import os

def generate_ens_ids(c, usage='full'):
    dataset = c['dataset']
    mart_export_df = pd.read_csv(c['filenames']['mart-export'], sep='\t')
    ens_ids = set(mart_export_df['Ensembl Gene ID'].values.tolist())
    if c['manual-annotated']:
        annotated_df = pd.read_csv(os.path.join(c['spreadsheets_dir'], f'suppl_{usage}_{dataset}_unique_gene.tsv'), sep='\t')
    else:
        if c['manual-interaction']:
            annotated_df = pd.read_csv(os.path.join(c['spreadsheets_dir'], f'automated_manually_curated_{c["dataset"]}_ens_ids_interactions.tsv'), sep='\t')
        else:
            annotated_df = pd.read_csv(os.path.join(c['spreadsheets_dir'], f'automated_{c["dataset"]}_ens_ids_interactions.tsv'), sep='\t')

    yes_ens_ids = set(annotated_df['Ensembl ID'].values.tolist())
    
    no_ens_ids = list(ens_ids - yes_ens_ids)
    yes_no_ens_ids = list(yes_ens_ids) + no_ens_ids
    
    yes_no = ["Yes"] * len(yes_ens_ids) + ["No"] * len(no_ens_ids)
    
    yes_no_df = pd.DataFrame()
    yes_no_df['Ensembl ID'] = yes_no_ens_ids
    yes_no_df['Interaction'] = yes_no
    
    if c['manual-annotated']:
        yes_no_df.to_csv(os.path.join(c['spreadsheets_dir'], f'manual_curated_{usage}_{dataset}_ens_ids_interactions.tsv'), sep='\t')
    else:
        if c['manual-interaction']:
            yes_no_df.to_csv(os.path.join(c['spreadsheets_dir'], f'automated_curated_{dataset}_ens_ids_interactions.tsv'), sep='\t')
        else:
            yes_no_df.to_csv(os.path.join(c['spreadsheets_dir'], f'automated_noncurated_{dataset}_ens_ids_interactions.tsv'), sep='\t')
        
        