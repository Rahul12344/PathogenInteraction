#!/bin/bash
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                     #-- the shell for the job
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l h_data=4G                  #-- submits on nodes with enough free memory (required)
#$ -l arch=linux-x64               #-- SGE resources (CPU type)
#$ -l h_rt=23:00:00               #-- runtime limit (see above; this requests 24 hours)
#$ -t 1-4480                        #-- remove first '#' to specify the number of
#$ -pe shared 4
#$ -N hyperparameter_tuning

readarray files < ./configs_rahul_protein_interaction/config_rahul_VIP

files=(null ${files[@]}) # this pads the file with an extra line in the beginning. 
file=${files[$SGE_TASK_ID]}
#file=${files[1]}
echo $file

/u/home/r/rahul/.conda/envs/grid_search_2/bin/python3.9 ./grid_search_02212022.py -outFile ./grid_search/grid_search_output_02162022.csv -configFile $file