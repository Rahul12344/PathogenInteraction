# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=11850928,11482001

import requests
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def convert_pub_date(pub_date):
    items = pub_date.split()
    converted_date = items[0]
    if len(items) > 1:
        if items[1][0:3] == "Jan":
            converted_date += "01"
        elif items[1][0:3] == "Feb":
            converted_date += "02"
        elif items[1][0:3] == "Mar":
            converted_date += "03"
        elif items[1][0:3] == "Apr":
            converted_date += "04"
        elif items[1][0:3] == "May":
            converted_date += "05"
        elif items[1][0:3] == "Jun":
            converted_date += "06"
        elif items[1][0:3] == "Jul":
            converted_date += "07"
        elif items[1][0:3] == "Aug":
            converted_date += "08"
        elif items[1][0:3] == "Sep":
            converted_date += "09"
        elif items[1][0:3] == "Oct":
            converted_date += "10"
        elif items[1][0:3] == "Nov":
            converted_date += "11"
        elif items[1][0:3] == "Dec":
            converted_date += "12"
        else:
            converted_date += "00"
    else:
        converted_date += "00"
    return int(converted_date)

def get_uids(partial=True):
    if partial:
        path_to_data = os.path.join("/Users/rahulnatarajan/Research/", 'Garud/train/dataset/')
        uids = []
        for category in ['pos', 'neg']:
            training_data = "updated_bacteria_train"
            validation_data = "updated_bacteria_test"
            testing_data = "updated_bacteria_validation"
            train_path = os.path.join(path_to_data, training_data, category)
            validation_path = os.path.join(path_to_data, validation_data, category)
            test_path = os.path.join(path_to_data, testing_data, category)
            for fname in sorted(os.listdir(train_path)):
                uids.append(fname.strip(".txt"))
            for fname in sorted(os.listdir(validation_path)):
                uids.append(fname.strip(".txt"))
            for fname in sorted(os.listdir(test_path)):
                uids.append(fname.strip(".txt"))
        return uids
    else:
        f = open("/Users/rahulnatarajan/Research/Garud/pubmedids_for_bacteria.txt")
        uids = []
        for line in f.readlines():
            uids.append(line)
        return uids

def create_query(uids):
    uid_str = ','.join(uids)
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={}".format(uid_str)

def request(query):
    return requests.get(query)

def parse_output(request_output):
    bp_data = []
    try:
        root = ET.fromstring(request_output.content)
        for DocSum in root.findall('DocSum'):
            published_date = DocSum.findall(".//*[@Name='PubDate']")[0].text
            bp_data.append(convert_pub_date(published_date))
    except:
        print("err")
    return bp_data
        
        
if __name__ == "__main__":
    uids = get_uids()
    bp_data = []
    for i in range(int(len(uids)/20)):
        curr = i*20     
        request_output = request(create_query(uids[curr:curr+20]))
        bp_data.extend(parse_output(request_output))
    bp_data.sort()
    
    total_uids = get_uids(partial=False)
    total_bp_data = []
    for i in range(int(len(uids)/20)):
        curr = i*20     
        request_output = request(create_query(total_uids[curr:curr+20]))
        total_bp_data.extend(parse_output(request_output))
    
    fig, ax = plt.subplots() 
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Date Ranges (YearMonth)')     
    bp = ax.boxplot([bp_data, total_bp_data], patch_artist=True, positions=[1, 1.4])
    
    """data = [bp_data, total_bp_data]
    for i in range(len(data)):
        x = np.random.normal(i, 0.04, size=len(data[i]))
        ax.scatter([a+1 for a in x], data[i], color='orange', zorder=100)"""
        
    i = 0
    for box in bp['boxes']:
        box.set(color='black', linewidth=2)
        box.set(facecolor = 'olive' )
        i+=1
    plt.xticks([1, 1.4], ["Prediction Set", "Total Set"])
    plt.xticks(rotation=90)
    plt.show()