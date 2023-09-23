import os

def create_mal_bac_diff(bact_file, mal_file):
    bact_file = open(bact_file)
    mal_file = open(mal_file)
    diff_file = open("outputs/diff_file.txt", "w")
    
    bact_file_lines = bact_file.readlines()
    mal_file_lines = mal_file.readlines()
    
    for i in range(0, len(bact_file_lines)):
        b_line = bact_file_lines[i].split()
        print(float(b_line[1]))
        m_line = mal_file_lines[i].split()
        diff_file.write("{0},{1},{2}\n".format(b_line[0], round(float(b_line[1])), round(float(m_line[1]))))
    
    bact_file.close()
    mal_file.close()
    diff_file.close()
    
def create_diff(rahul_file, anshu_file):
    r_file = open(rahul_file)
    a_file = open(anshu_file)
    diff_file = open("outputs/diff.txt", "w")
    
    r_file_lines = r_file.readlines()
    a_file_lines = a_file.readlines()
    
    for i in range(1, len(r_file_lines)):
        r_line = r_file_lines[i].rstrip().split(',')
        a_line = a_file_lines[i].rstrip().split(',')
        diff_file.write("{0},{1},{2},{3},{4}\n".format(r_line[0], r_line[1], a_line[0], r_line[2], a_line[1]))
    
    r_file.close()
    a_file.close()
    diff_file.close()
    
def create_unified(data_path, positive_percentage=1.0, seed=123, dataset_size=1000):
    pos_size = float(dataset_size) * positive_percentage
    neg_size = dataset_size-pos_size
    
    path_to_data = os.path.join(data_path, 'Garud')
    train_texts = []
    train_labels = []
    training_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_training', "pos")))
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, 'malaria_training', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    sentence = ' '.join(filtered_sentence)
                    f = open("/Users/rahulnatarajan/Research/src/train/unified_malaria_train/{}/{}".format(category,fname), "w")
                    f.write(sentence)
                    f.close()
                
    val_texts = []
    val_labels = []
    validation_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_validation', "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'malaria_validation', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    sentence = ' '.join(filtered_sentence)
                    f = open("/Users/rahulnatarajan/Research/Garud/src/train/unified_malaria_validation/{}/{}".format(category,fname), "w")
                    f.write(sentence)
                    f.close()
                
                
    test_texts = []
    test_labels = []
    testing_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_testing', "pos")))
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'malaria_testing', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    sentence = ' '.join(filtered_sentence)
                    f = open("/Users/rahulnatarajan/Research/Garud/src/train/unified_malaria_test/{}/{}".format(category,fname), "w")
                    f.write(sentence)
                    f.close()

def overlap_comparison():
    f1 = open("/Users/rahulnatarajan/Research/Garud/src/train/outputs/bacttestclassifcation.txt", "r")
    f2 = open("/Users/rahulnatarajan/Research/Garud/src/train/outputs/bacttestclassifcation_malaria.txt", "r")
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    file_path = open("unity.txt", "w")
    for i in range(len(lines1)):
        line1 = lines1[i].split()
        line2 = lines2[i].split()
        if line1[1] == '1.000000' and line2[1] == '1.000000':
            file_path.write("{}\n".format(line1[0]))
    file_path.close()
    return
    
              
    

