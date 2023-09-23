import pandas as pd
from tqdm import tqdm

def get_interaction_item_from_hgncs(c, hgncs, gene_aliases):
    if c['dataset'] == "VIP":
        df = pd.read_csv(c['filenames']['virus-annotation'], sep='\t')
    else:
        df = pd.read_csv(c['filenames']['bacteria-annotation'], sep='\t')
    
    interaction_items = list()
    
    
    for hgnc in tqdm(hgncs):
        alias_loc = find_gene(hgnc, gene_aliases)
        gene_list = gene_aliases[alias_loc]
        found = False
        for i in tqdm(range(len(df))):
            if not pd.isna(df.iloc[i]['Gene(s)']):
                df_iloc_genes = df.iloc[i]['Gene(s)'].upper()
                for gene in gene_list:
                        gene = " {} ".format(gene)
                        dash_less_gene = " {} ".format(gene).replace('-', '')
                        dash_less_df_iloc_genes = " {} ".format(df_iloc_genes).replace('-', '')
                        if gene != ' A ':
                            if gene in df_iloc_genes or dash_less_gene in df_iloc_genes or gene in dash_less_df_iloc_genes or dash_less_gene in dash_less_df_iloc_genes:
                                if c['dataset'] == "VIP":
                                    interaction_item = df.iloc[i]['Virus(es)']
                                else:
                                    interaction_item = df.iloc[i]['Species(s)']
                                interaction_items.append(interaction_item.lower())
                                found = True
                                break 
                if found:
                    break      
        if not found:
            interaction_items.append('')
    return interaction_items

def get_ens_id_list(content, ens_ids, aliases_list):    
    ens_id_output = list()
    for _, line in enumerate(content):
        for curr_num_alias, aliases in enumerate(aliases_list):
            for gene in aliases:
                normalized_line = line.upper().rstrip('\n').replace('-', '')
                if gene != '\n':
                    gene = gene.rstrip('\n')
                    if normalized_line.find(gene) != -1 and len(gene) > 2:
                        ens_id_output.append([ens_ids[curr_num_alias]])
                        break
    return ens_id_output
    
def find_gene(gene, aliases):
    for line_num in range(len(aliases)):
        if gene in aliases[line_num]:
            return line_num
        
def get_gene_aliases(c):
    aliases = list()
    with open(c['filenames']['gene-aliases'], 'r') as f:
        content = f.readlines()
    for line in content:
        aliases.append(line.split('\t'))
    
    return aliases

    
def get_ens_id_list(c):
    with open(c['filenames']['mart-export'], 'r') as f:
        content = f.readlines()
        ens_id = []
    
    for _,line in enumerate(content[1:]):
        split_line = line.split()
        ens_id.append(split_line[0])
 
    return ens_id

def spreadsheet(c):
    # Read in the data
    aliases = get_gene_aliases(c)
    
    if c['dataset'] == "VIP":
        df = pd.read_csv(c['filenames']['virus-interaction'], sep='\t')
    else:
        df = pd.read_csv(c['filenames']['bacteria-interaction'], sep='\t')
      
    yes_df = df[df['Interaction'] == 'Yes']
    ens_ids = yes_df['ID'].tolist()
    
    mart_export_df = pd.read_csv(c['filenames']['mart-export'], sep='\t')
    HGNC_sym = []
    for i in ens_ids:
        HGNC_sym.append(mart_export_df[mart_export_df['Ensembl Gene ID'] == i]['HGNC symbol'].tolist()[0])
    
    interactions = get_interaction_item_from_hgncs(c, HGNC_sym, aliases)
    
    table_df = pd.DataFrame()    
    table_df['Ensembl ID'] = ens_ids
    table_df['HGNC Symbol'] = HGNC_sym
    host_species = ['human']*len(ens_ids)
    table_df['Host Species'] = host_species
    
    if c['dataset'] == "VIP":
        table_df['Virus Name'] = interactions
    else:
        table_df['Bacteria Name'] = interactions
    
    dataset = c['dataset']
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f'suppl_{dataset}.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    table_df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    