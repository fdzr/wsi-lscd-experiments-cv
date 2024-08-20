#!/bin/bash

methods="chinese_whispers correlation_clustering wsbm spectral_clustering"

for m in $methods
do
    for index in $(seq 1 5)
    do
        mkdir -p missing-results/cv-experiments/$m/${index}_fold
    done
done