def find_overlap(dataset_one, dataset_two):
    overlap_abstract_list = []
    overlap_gene_list = []
    
    dataset_one_content = set()
    dataset_two_content = set()
    
    if dataset_one == "old_vip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/VIPs_PMID_for_Rahul.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            dataset_one_content.add(line.split("\t")[0])
        read_file.close()
    elif dataset_one == "new_vip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/virus_interaction.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            if split_line[1] == "Yes":
                dataset_one_content.add(split_line[0])
        read_file.close()
    elif dataset_one == "vip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/VIPs_PMID_for_Rahul.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            dataset_one_content.add(line.split("\t")[0])
        read_file.close()
        read_file = open("/Users/rahulnatarajan/Research/Garud/virus_interaction.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            if split_line[1] == "Yes":
                dataset_one_content.add(split_line[0])
        read_file.close()
    elif dataset_one == "bip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/bacteria_interaction.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            if split_line[1] == "Yes":
                dataset_one_content.add(split_line[0])
        read_file.close()
    elif dataset_one == "pip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/journal.pgen.1007023.s006.csv", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            dataset_one_content.add(line.split(",")[0])
        read_file.close()
    
    if dataset_two == "old_vip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/VIPs_PMID_for_Rahul.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            dataset_two_content.add(line.split("\t")[0])
        read_file.close()
    elif dataset_two == "new_vip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/virus_interaction.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            if split_line[1] == "Yes":
                dataset_two_content.add(split_line[0])
        read_file.close()
    elif dataset_two == "vip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/VIPs_PMID_for_Rahul.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            dataset_two_content.add(line.split("\t")[0])
        read_file.close()
        read_file = open("/Users/rahulnatarajan/Research/Garud/virus_interaction.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            if split_line[1] == "Yes":
                dataset_two_content.add(split_line[0])
        read_file.close()
    elif dataset_two == "bip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/bacteria_interaction.txt", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            if split_line[1] == "Yes":
                dataset_two_content.add(split_line[0])
        read_file.close()
    elif dataset_two == "pip":
        read_file = open("/Users/rahulnatarajan/Research/Garud/journal.pgen.1007023.s006.csv", "r")
        lines = read_file.readlines()
        for line in lines[1:]:
            dataset_two_content.add(line.split(",")[0])
        read_file.close()
    
    for ens_id in dataset_one_content:
        if ens_id in dataset_two_content:
            overlap_gene_list.append(ens_id)
    print(dataset_one_content)
    #output_file_abstract = dataset_one + "_" + dataset_two + "_abstract_overlap"
    output_file_gene = dataset_one + "_" + dataset_two + "_gene_overlap"
    #output_file_abstract_write = open(output_file_abstract, "w")
    output_file_gene_write = open(output_file_gene, "w")
    #for overlap in overlap_abstract_list:
    #    output_file_abstract_write.write(overlap + "\n")
    #output_file_abstract_write.close()
    for overlap in overlap_gene_list:
        output_file_gene_write.write(overlap + "\n")
    output_file_gene_write.close()
    return

if __name__ == '__main__':
    find_overlap("pip", "bip")