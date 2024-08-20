#!/bin/bash

# wget https://zenodo.org/record/8197553/files/dwug_de_sense.zip && unzip dwug_de_sense.zip

methods="spectral_clustering-llama"
kfold=5
cv=1

for m in $methods
do
    if [ $cv -eq 1 ]
    then
        for index in $(seq 1 $kfold)
        do
            mkdir -p cv-experiments/$m/${index}_fold
            python3 generate_result_files.py \
                -p cv-experiments/$m/${index}_fold/training.csv \
                -m $m \
                -d dwug_data_annotated_only
            python3 generate_result_files.py \
                -p cv-experiments/$m/${index}_fold/testing.csv \
                -m $m \
                -d dwug_data_annotated_only
        done
    else
        mkdir -p no-cv-experiments-$m/$m/all_words
        python3 generate_result_files.py \
            -p no-cv-experiments-$m/$m/all_words/training.csv \
            -m $m \
            -d dwug_data_annotated_only

    fi
done

# jupyter nbconvert --to notebook --execute base-code/Chinese_Whispers.ipynb &
# jupyter nbconvert --to notebook --execute base-code/Correlation_clustering.ipynb &
# # # jupyter nbconvert --to notebook --execute base-code/WSBM.ipynb &
# jupyter nbconvert --to notebook --execute base-code/Spectral_clustering.ipynb &

# wait