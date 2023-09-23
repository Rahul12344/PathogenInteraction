from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def write_go_dict(go_dict):
    write_file = open("go_dict_manual.txt", 'w')
    for entry in go_dict:
        str_values = ['\t'.join(x) for x in go_dict[entry]]
        write_file.write("sample_{}\n{}\n".format(entry, '\t'.join(str_values)))
    write_file.close()

def create_go_for_all_samples():
    ens_to_hgnc_dict = ens_to_hgnc()
    
    ontology_io = open("/Users/rahulnatarajan/Research/Garud/go.txt", "r")
    content = ontology_io.read()
    ontology_io.close()
    gene_ontology = parse_gene_ontology(content)
    
    annotations_io = open("/Users/rahulnatarajan/Research/Garud/goa_human.gaf", "r")
    content = annotations_io.read()
    annotations_io.close()
    annotations, annotations_hgnc = parse_gene_annotations(content)
    
    go_dict = defaultdict(list)
    
    samples_file = open("/Users/rahulnatarajan/Research/all_nonBIPs.txt", "r")
    text = samples_file.read()
    samples_file.close()
    
    samples = defaultdict(list)
    
    sample_list = text.split("sample")
    counter = 338
    for item in sample_list[338:]:
        item = item.replace('\n', ' ')
        items = item.split(' ')
        samples[counter] = items[1:len(items)-1]
        counter += 1
      
    write_file = open("go_dict_manual_337_.txt", 'w')      
    for sample in samples:
        ens_ids = samples[sample]
        go_dict[sample] = create_go_dicts(ens_ids, ens_to_hgnc_dict, annotations_hgnc, gene_ontology)
        write_file.write("sample_{}\n{}\n\n".format(sample, '\t'.join(go_dict[sample])))
        
    write_file.close()

def create_go_dicts(ens_ids, ens_to_hgnc_dict, annotations_hgnc, gene_ontology):
    superset_list = []
    for ens_id in ens_ids:
        hgnc = convert_ens_to_hgnc(ens_id, ens_to_hgnc_dict)
        superset = set()
        if hgnc != "Not Found":
            go_ids = convert_hgnc_to_go(hgnc, annotations_hgnc)
            if go_ids != []:
                for go_id in go_ids:
                    superset = superset.union(explore_full_depth(go_id, set(), gene_ontology))
        superset_list.extend(list(superset))
    return superset_list
  
def parse_gene_ontology(content):
    ontology_entries = content.split('\n\n')
    ontology_dictionary = []
    for entry in ontology_entries:
        curr_dict = defaultdict(list)
        entry_info = entry.split('\n')[1:]
        for info in entry_info:
            split_info = info.split(': ')
            if len(split_info) > 1:
                if split_info[0] == "is_a":
                    curr_dict[split_info[0]].append(split_info[1][0:10])
                else:
                    curr_dict[split_info[0]].append(split_info[1].rstrip('\n'))
        if curr_dict:
            ontology_dictionary.append(curr_dict)
    return ontology_dictionary

def parse_gene_annotations(content):
    annotations_entries = content.split('\n')[41:]
    annotations_dictionary = defaultdict(list)
    annotations_hgnc_dictionary = defaultdict(list)
    for entry in annotations_entries:
        entry_info = entry.split('\t')
        go_id = entry_info[4]
        annotations_dictionary[entry_info[1]].append(go_id)
        annotations_hgnc_dictionary[entry_info[2]].append(go_id)

    return annotations_dictionary, annotations_hgnc_dictionary

def get_ens_ids():
    ens_id_interactions_io = open("/Users/rahulnatarajan/Research/Garud/bacteria_interaction.txt", "r")
    ens_interactions = []
    
    lines = ens_id_interactions_io.readlines()
    for line in lines:
        split_line = line.split(' ')
        if split_line[1] == "Yes\n":
            ens_interactions.append(split_line[0])
    
    ens_id_interactions_io.close()
    return ens_interactions

def ens_to_hgnc():
    mart_export_io = open("/Users/rahulnatarajan/Research/Garud/mart_export.txt", "r")
    ens_hgnc_lines = mart_export_io.readlines()[1:]
    ens_to_hgnc_dict = {}
    for ens_hgnc_line in ens_hgnc_lines:
        split_line = ens_hgnc_line.split('\t')
        if len(split_line) > 1:
            ens_to_hgnc_dict[split_line[0]] = split_line[1].rstrip('\n')
    return ens_to_hgnc_dict

def convert_ens_to_hgnc(ens_id, ens_to_hgnc):
    if ens_id in ens_to_hgnc:
        return ens_to_hgnc[ens_id]
    return "Not Found"

def convert_ens_to_uni(ens_id, ens_to_uni):
    return ens_to_uni[ens_id]

def convert_hgnc_to_go(hgnc_id, hgnc_to_go):
    if hgnc_id in hgnc_to_go:
        return hgnc_to_go[hgnc_id]
    return []

def convert_uni_to_go(uni_id, uni_to_go):
    if uni_id in uni_to_go:
        return uni_to_go[uni_id]
    return "Not Found"
    
def explore_full_depth(id, curr_set, gene_ontology):
    info = get_info(id, gene_ontology)
    curr_set.add(id)
    is_as = get_is_as(info)
    if is_as == []:
        return curr_set
    for is_a in is_as:
        if is_a not in curr_set:
            explore_full_depth(is_a, curr_set, gene_ontology)
            
    return curr_set

def get_info(id, gene_ontology):
    for entry in gene_ontology:
        if "id" in entry and entry["id"][0] == id:
            return entry
    return None

def get_is_as(info):
    if info != None and "is_a" in info:
        return info["is_a"]
    return []   

def enrichment_and_p_values():
    bacteria_go_count_io = open("/Users/rahulnatarajan/Research/Garud/compose_go_symbols.tsv", "r")
    bacteria_go_information = bacteria_go_count_io.readlines()
    bacteria_go_count_io.close()
    
    bacteria_go_count = defaultdict(int)
    
    for bacteria_go_info in bacteria_go_information:
        go_information = bacteria_go_info.split('\t')[2:]
        for go_info in go_information:
            bacteria_go_count[go_info] += 1
    
    control_go_count_io = open("/Users/rahulnatarajan/Research/Garud/go_dict_manual.txt", "r")
    total_control = control_go_count_io.read().rstrip('\n')
    total_control_lines = total_control.split('\n\n')
    control_go_count_io.close()
    
    go_list = []
    
    for line in total_control_lines:
        line_dict = defaultdict(float)

        for go in line.split('\n')[1].split('\t'):
            line_dict[go] += 1.0
        go_list.append(line_dict)
    
    control_go_count_io = open("/Users/rahulnatarajan/Research/Garud/go_dict_manual_337_.txt", "r")
    total_control = control_go_count_io.read().rstrip('\n')
    total_control_lines = total_control.split('\n\n')
    control_go_count_io.close()
    
    for line in total_control_lines:
        line_dict = defaultdict(float)
        for go in line.split('\n')[1].split('\t'):
            line_dict[go] += 1.0
        go_list.append(line_dict)
    
    enrichment_dict = defaultdict(float)
    p_value_dict = defaultdict(float)
    for go_category in bacteria_go_count:
        for go_dict in go_list:
            if go_category not in go_dict:
                go_val = bacteria_go_count[go_category]
                enrichment_dict[go_category] += go_val + 1.0
            else:
                enrichment_dict[go_category] += (bacteria_go_count[go_category]/go_dict[go_category])
    count = 0
    for go_category in enrichment_dict:
        enrichment_dict[go_category] /= 1000
 
    
    for go_dict in go_list:
        for go_category in go_dict:
            if go_category not in bacteria_go_count:
                go_val = go_dict[go_category]
                p_value_dict[go_category] += 1.0
            else:
                if go_dict[go_category] > bacteria_go_count[go_category]:
                    p_value_dict[go_category] += 1.0
    
    for go_category in bacteria_go_count:
        if go_category not in p_value_dict:
            p_value_dict[go_category] = 0.0
     
    for go_category in p_value_dict:
        p_value_dict[go_category] /= 1000 
        
    
        
    print(len(p_value_dict))

    enriched_go_categories = []
    
    enrichment_file_io = open("/Users/rahulnatarajan/Research/enrichment_go_values.tsv", "w")
   
    """
    for go_category in enrichment_dict:
        enrichment_file_io.write("{}\t{}\n".format(go_category, enrichment_dict[go_category]))
    
    enrichment_file_io.close()
    """
    p_value_file_io = open("/Users/rahulnatarajan/Research/Garud/p_value_go_values.tsv", "w")
    
    for go_category in p_value_dict:
        p_value_file_io.write("{}\t{}\n".format(go_category, p_value_dict[go_category]))
    
    p_value_file_io.close()
    
    
    for go_category in enrichment_dict:
        if enrichment_dict[go_category] > 1.0 and p_value_dict[go_category] < 0.05:
            enriched_go_categories.append((go_category, enrichment_dict[go_category],  p_value_dict[go_category]))
    
    
    expr_lymph_io = open("/Users/rahulnatarajan/Research/expr_lympho.txt", "r")
    expr_lymph_dict = defaultdict(float)
    lines = expr_lymph_io.readlines()
    expr_lymph_io.close()
    
    for line in lines:
        info = line.split(" ")
        expr_lymph_dict[info[0]] = float(info[1])
    
    
    bacteria_gene_io = open("/Users/rahulnatarajan/Research/all_BIPs.txt", "r")
    bacteria_genes = []
    lines = bacteria_gene_io.readlines()
    bacteria_gene_io.close()   
    for line in lines:
        bacteria_genes.append(line.rstrip('\n'))
        
    samples_file = open("/Users/rahulnatarajan/Research/all_nonBIPs.txt", "r")
    text = samples_file.read()
    samples_file.close()
    
    samples = []
    
    sample_list = text.split("sample")
    for item in sample_list[1:]:
        item = item.replace('\n', ' ').rstrip(' ')
        items = item.split(' ')
        samples.append(items[1:len(items)-1])
    
    start = 0.0
    for gene in bacteria_genes:
        start += expr_lymph_dict[gene]
    
    means = []   
    for sample in samples:
        temp = 0.0
        for gene in sample:
            temp += expr_lymph_dict[gene]
        means.append(temp/1000)

    ave_start = start/1000
    return enriched_go_categories, means, ave_start

if __name__ == '__main__':
    #create_go_for_all_samples()
    enriched_go_categories, means, ave_start = enrichment_and_p_values()
    
    """
    enriched_file_io = open("/Users/rahulnatarajan/Research/Garud/enriched_go_categories.tsv", "w")
    for go, enrich, p_val in enriched_go_categories:
        enriched_file_io.write("{}\t{}\t{}\n".format(go, enrich, p_val))
        
    enriched_file_io.close()
    
    max_mean = max(means)
    min_mean = min(means)
    
    bucket_size = (max_mean - min_mean)/15.0
    buckets = [[]]*15
    print(buckets)
    indices = []
    
    for i in range(16):
        indices.append(min_mean + bucket_size*(i))
    
    for mean in means:
        for i in range(15):
            if mean <= min_mean + bucket_size*(i+1) and mean >= min_mean + bucket_size*(i):
                buckets[i].append(mean) 
    
    plt.hist(buckets, indices, color=[(1.0, 0.5, 0.0, 0.6)]*15)
    plt.axvline(x=ave_start, label= 'Average GTEX Enrichment Value in BIPs')
    plt.legend(loc = 'upper right')
    plt.ylabel("Frequency of GTEX Mean Enrichment Value", rotation="vertical")
    plt.xlabel("GTEX Mean Enchrichment Value", rotation="horizontal")
    plt.show()  
    """