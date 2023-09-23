from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def build_bacteria_distribution(input_file):
    read_input_file = open(input_file, "r")
    
    lines = read_input_file.readlines()[1:]
    
    read_input_file.close()
    
    bacteria_count = defaultdict(int)
    
    for line in lines:
        bacteria = line.split('\t')[4].lower().rstrip()
        bacteria = bacteria.replace("and", ",")
        bacteria_split = bacteria.split(',')
        for bacterium in bacteria_split:
            if bacterium != "":
                if bacterium.find("pylori") != -1:
                    bacteria_count["h. pylori"] += 1
                elif bacterium.find("salmonella") != -1:
                    bacteria_count["salmonella"] += 1
                else:
                    bacteria_count[bacterium] += 1
        
    bacteria = bacteria_count.keys()
    counts = []
    for bacterium in bacteria:
        counts.append(bacteria_count[bacterium])
    items = list(zip(bacteria, counts))
    items = sorted(items, 
       key=lambda x: x[1])
    items.reverse()
    bacteria, num = [[i for i, j in items ],
       [j for i, j in items ]]
    plt.bar(bacteria, num, color=[(1.0, 0.5, 0.0, 0.6)]*len(bacteria),  edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel("Number of BIPs Containing Bacteria", rotation="vertical")
    plt.xlabel("Bacteria Name", rotation="horizontal")
    plt.show()  
    
if __name__ == '__main__':
    build_bacteria_distribution("/Users/rahulnatarajan/Research/Garud/positive_bacteria_tsv.tsv")