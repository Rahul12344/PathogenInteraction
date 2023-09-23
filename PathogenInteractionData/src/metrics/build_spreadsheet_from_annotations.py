import pandas as pd
from tqdm import tqdm
import datetime
import os

def get_interaction_item_from_hgncs(c):
    if c['dataset'] == "VIP":
        df = pd.read_csv(c['filenames']['virus-annotation'], sep='\t')
    else:
        df = pd.read_csv(c['filenames']['bacteria-annotation'], sep='\t')
    
    return df


    
def get_ens_id_list(c):
    mart_export_df = pd.read_csv(c['filenames']['mart-export-dashless'], sep='\t')
    return mart_export_df

def spreadsheet(c, usage='full', const_interaction_organism='human'):
    # Read in the data
    mart_export_df = get_ens_id_list(c)
    df = get_interaction_item_from_hgncs(c)
    if usage == 'partial':
        df = df[:189]
    df = df[df['HUGO'].notna()]
    
    ens_ids = []
    genes = []
    interaction_organisms = []
    interactions = []
    
    for i, row in tqdm(df.iterrows()):
        if c['dataset'] == "virus":
            interaction_items = row['Virus(es)']
        if c['dataset'] == "bacteria":
            interaction_items = row['Species(s)']
        interaction_items = [interaction_item.strip() for interaction_item in interaction_items.split(',')]
        
        hgnc_symbols = [hgnc.strip() for hgnc in row['HUGO'].split(',')]
        for hgnc_symbol in hgnc_symbols:
            matching_ens_ids = mart_export_df[mart_export_df['HGNC symbol'] == hgnc_symbol]['Ensembl Gene ID']
            if len(matching_ens_ids) > 0:
                ens_id = matching_ens_ids.values[0]
                for interaction_item in interaction_items:
                    ens_ids.append(ens_id)
                    genes.append(hgnc_symbol)
                    interaction_organisms.append(const_interaction_organism)
                    interactions.append(interaction_item)
            
        
    table_df = pd.DataFrame()    
    table_df['Ensembl ID'] = ens_ids
    table_df['HGNC Symbol'] = genes
    table_df['Host Species'] = interaction_organisms
    
    if c['dataset'] == "virus":
        table_df['Virus Name'] = interactions
    if c['dataset'] == "bacteria":
        table_df['Bacteria Name'] = interactions
    
    dataset = c['dataset']
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join(c['spreadsheets_dir'], f'suppl_{usage}_{dataset}_{current_date}.xlsx'), engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    table_df = table_df.drop_duplicates()
    table_df.to_csv(os.path.join(c['spreadsheets_dir'], f'suppl_{usage}_{dataset}.tsv'), sep='\t')
    
    table_df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    hngc_symbols = table_df['HGNC Symbol'].unique().tolist()
    rows = []
    for hgnc_symbol in hngc_symbols:
        matching_rows = table_df[table_df['HGNC Symbol'] == hgnc_symbol]
        ens_id = matching_rows['Ensembl ID'].unique().tolist()[0]
        bacteria = ','.join(matching_rows['Bacteria Name'].unique().tolist())
        rows.append((hgnc_symbol, ens_id, const_interaction_organism, bacteria))
    col_name = c["dataset"].capitalize()
    together_df = pd.DataFrame(rows, columns=['HGNC Symbol', 'Ensembl ID', 'Host Species', f'{col_name} Name'])
    together_df.to_csv(os.path.join(c['spreadsheets_dir'], f'suppl_{usage}_{dataset}_unique_gene.tsv'), sep='\t')
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(os.path.join(c['spreadsheets_dir'], f'suppl_{usage}_{dataset}_{current_date}_unique_gene.xlsx'), engine='xlsxwriter')
    
    # Convert the dataframe to an XlsxWriter Excel object.
    together_df.to_excel(writer, sheet_name='Sheet1')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()