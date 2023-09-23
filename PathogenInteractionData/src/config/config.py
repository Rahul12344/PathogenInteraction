import os
import yaml

def config():
        # load config from config.yaml
    with open('src/config/config.yaml', 'r') as f:
        c = yaml.safe_load(f)
    # init with config from utils/config.yaml
    
    env_usr = os.environ.get("USER")
    c['aws']= True if "ubuntu" in env_usr else False
    c['data-directory']=c['data-directory-ec2'] if c['aws'] else c['data-directory-local']  
        
    # for every key in config that ends with 'dir', add the data directory to the beginning
    for key in c:
        if key.endswith('dir'):
            c[key] = os.path.join(c['data-directory'], c[key])
            os.makedirs(c[key], exist_ok=True) # create the directory if it doesn't exist
    
    # alias_results    OR  gene_and_aliases
    c['filenames']={    
        'mart-export':         os.path.join(c['data-directory'], 'mart_export.tsv'), # this is the file that contains the gene names and their ensembl IDs
        'mart-export-dashless':         os.path.join(c['data-directory'], 'mart_export_dashless.tsv'), # this is the file that contains the gene names and their ensembl IDs
        'gene-aliases':        os.path.join(c['annotation_dir'], 'alias_results.tsv'), # this is the file that contains the gene names and their aliases
        'abstract-genelist':   os.path.join(c['annotation_dir'], 'abstract_gene_bacteria.tsv'),
        'virus-interaction':   os.path.join(c['annotation_dir'], 'virus_interaction.tsv'), 
        'bacteria-interaction':os.path.join(c['annotation_dir'], 'bacteria_interaction.tsv'),
        'bacteria-annotation': os.path.join(c['annotation_dir'], 'positive_bacteria_annotations.tsv'),
        'virus-annotation':    os.path.join(c['annotation_dir'], 'positive_virus_annotations.tsv'),
        'enriched-ids':     os.path.join(c['data-directory'], 'mass-spec_VIPs.txt'),
        'virus-ids':        os.path.join(c['data-directory'], 'VIPs_PMID.txt'),
        'malaria-ids':      os.path.join(c['data-directory'], 'journal.pgen.1007023.s006.csv'),
    }
    
    if c['dataset'] == 'bacteria':
        c['model_data_dir'] = c['bacteria_dir']
    elif c['dataset'] == 'virus':
        c['model_data_dir'] = c['virus_dir']
    elif c['dataset'] == 'malaria':
        c['model_data_dir'] = c['malaria_dir']
        
    if c['dataset'] == 'bacteria':
        c['download_dir'] = os.path.join(c['bacteria_dir'], 'text')
    elif c['dataset'] == 'virus':
        c['download_dir'] = os.path.join(c['virus_dir'], 'text')
    elif c['dataset'] == 'malaria':
        c['download_dir'] = os.path.join(c['malaria_dir'], 'text')
    elif c['dataset'] == 'new_bacteria':
        c['download_dir'] = os.path.join(c['bacteria_dir'], 'new')
    elif c['dataset'] == 'new_virus':
        c['download_dir'] = os.path.join(c['virus_dir'], 'new')
    elif c['dataset'] == 'new_malaria':
        c['download_dir'] = os.path.join(c['malaria_dir'], 'new')
    elif c['dataset'] == 'recapture_virus':
        if not c['neg']:
            c['download_dir'] = os.path.join(c['virus_dir'], 'recapture')
        else:
            c['download_dir'] = os.path.join(c['virus_dir'], 'negrecapture')

    return c