import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import config
from metrics.build_spreadsheet_from_annotations import spreadsheet
from metrics.build_distribution import dist, generated_score_distributions
from metrics.generate_ens_id import generate_ens_ids
from metrics.automated_abstract_gene_annotation import automatically_annotate_abstracts
from utils.plot_util import set_plot_info

if __name__ == '__main__':
    c = config()
    
    set_plot_info(c)
    if c['spreadsheet']:
        print('Building spreadsheet...')
        spreadsheet(c)
    if c['build-interaction-distribution']:
        print('Building distribution...')
        dist(c)
    if c['automatically-annotate']:
        print('Building automatic annotations...')
        automatically_annotate_abstracts(c)
    if c['ens-id-output']:
        print('Building ensemble id output...')
        generate_ens_ids(c)
    if c['score-distribution']:
        print('Building score distribution...')
        generated_score_distributions(c)