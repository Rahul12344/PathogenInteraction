from tqdm import tqdm
import os
import pandas as pd


def automatically_annotate_abstracts(c):
    gene_interactions = []
    gene_alias_df = pd.read_csv(c['filenames']['gene-aliases'], sep='\t')
    mart_export_df = pd.read_csv(c['filenames']['mart-export-dashless'], sep='\t')
    abstracts = get_positively_scored_abstracts(c)
    for abstract in tqdm(abstracts[0:]):
        gene_interactions += find_gene_interaction(gene_alias_df, open_abstract_file(c, abstract))
    gene_interactions = list(set(gene_interactions))
    ens_ids = []
    hgncs = []
    for gene in tqdm(gene_interactions):
        ens_id = get_ens_ids(gene, mart_export_df)
        if len(ens_id) > 0:
            ens_ids.append(ens_id[0])
            hgncs.append(gene)
    df = pd.DataFrame()
    df['Ensembl ID'] = ens_ids
    df['HGNC symbols'] = hgncs
    
    if c['manual-interaction']:
        df.to_csv(os.path.join(c['spreadsheets_dir'], f'automated_manually_curated_{c["dataset"]}_ens_ids_interactions.tsv'), sep='\t')
    else:
        df.to_csv(os.path.join(c['spreadsheets_dir'], f'automated_{c["dataset"]}_ens_ids_interactions.tsv'), sep='\t')

def get_ens_ids(gene, mart_export_df):
    ens_ids = mart_export_df[mart_export_df['HGNC symbol'] == gene]['Ensembl Gene ID'].values.tolist()
    return ens_ids

def get_positively_scored_abstracts(c):
    if c['manual-interaction']:
        annotation_df = pd.read_csv(c['filenames']['bacteria-annotation'], sep='\t')
        
        annotation_df = annotation_df[annotation_df['Abstract'].notna()]
        annotation_df = annotation_df[annotation_df['Gene(s)'].notna()]
        abstracts = annotation_df['Abstract'].values.tolist()
        
        txt_abstracts = []
        for abstract in abstracts:
            if '.txt' not in abstract:
                txt_abstracts.append(abstract + '.txt')
            else:
                txt_abstracts.append(abstract)
            
        return txt_abstracts
    else:
        annotation_df = pd.read_csv(os.path.join(c['annotation_dir'], "bacteria_prediction.tsv"), sep='\t')
        cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        #keep abstracts that have a score of 1 for more then 5 of the columns
        pos_df = annotation_df[annotation_df[cols].sum(axis=1) > 5]
        abstracts = pos_df['Abstract'].values.tolist()
            
        return abstracts

def open_abstract_file(c, abstract_name):
    try:
        with open(os.path.join(c['bacteria_dir'], "new", abstract_name), "r") as abstract_file:
            lines = str(abstract_file.read())
        lines = lines.upper()
        lines = lines.rstrip('\n')
        lines = lines.replace('-', '')
        return lines
    except FileNotFoundError:
        return ''

# c['filenames']['gene-aliases']
def find_gene_interaction(gene_alias_df, lines):
    found_genes = []
    for _, row in gene_alias_df.iterrows():
        gene = row['Approved symbol']
        alias = row['Alias symbol']
        name = row['Approved name']
        if not pd.isna(name):
            name = name.upper()
        
        if f" {gene} " in lines:
            found_genes.append(gene)
        elif not pd.isna(alias) and f" {alias} " in lines:
            found_genes.append(gene)
        elif not pd.isna(name) and name in lines:
            found_genes.append(gene)
    return list(set(found_genes))

    