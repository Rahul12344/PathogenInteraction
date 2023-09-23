 # generate config file for VIP
 
min_ngram_size=1
max_ngram_size=5

min_learning_rate=1e-8
max_learning_rate=1e-2

min_layers=1
max_layers=4

min_dropout_rate=0.2
max_dropout_rate=0.5

min_units=8
max_units=64

ngram_sizes = list(range(min_ngram_size, max_ngram_size+1))
learning_rates = list()
curr_learn_rate = min_learning_rate
while curr_learn_rate <= max_learning_rate:
    learning_rates.append(curr_learn_rate)
    curr_learn_rate *= 10
layers = list(range(min_layers, max_layers+1))
dropout_rates = list()
curr_dropout_rate = min_dropout_rate
while curr_dropout_rate <= max_dropout_rate:
    dropout_rates.append(curr_dropout_rate)
    curr_dropout_rate += 0.1
units_per_layer = list(range(min_units, max_units+1, 8))

for ngram_size in ngram_sizes: 
    for learning_rate in learning_rates:
        for layer in layers:
            for dropout_rate in dropout_rates:
                for unit in units_per_layer:
                    f = open("./configs_rahul_protein_interaction/config_rahul_VIP", "a")
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(ngram_size, learning_rate, layer, dropout_rate, unit))
                    f.close()