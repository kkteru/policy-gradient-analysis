#!/bin/bash

mkdir -p results_larger_critic
mkdir -p results_critic_iterations
mkdir -p results_baseline

#echo "STARTING BASELINE"
#for ((i=0;i<5;i+=1))
#do
#    echo "BASELINE seed $i"
#    python main_baseline.py --env_name $1 --seed $i --use_logger True --name_prefix BASELINE --folder ./results_baseline >> results_baseline/$1.log 2>&1
#done

folder_name="results_larger_critic_with_5_repeats"
mkdir -p $folder_name
rm ./$folder_name/$1.log

echo "STARTING LARGE CRITIC"
for ((i=0;i<5;i+=1))
do
    echo "LARGE_CRITIC seed $i"
    python main_baseline.py --env_name $1 --seed $i --use_logger True --larger_critic_approximator True --critic_repeat 5 --folder ./$folder_name --name_prefix LARGE_CRITIC >> ./$folder_name/$1.log 2>&1
done

#declare -a iters=("3" "5" "7" "10")
#
#echo "STARTING CRITIC ITERATIONS"
#for iter in ${iters[@]}
#do
#    for ((i=0;i<5;i+=1))
#    do
#        echo "CRITIC ITERATIONS = $iter, seed $i"
#        python main_baseline.py --env_name $1 --seed $i --use_logger True --folder ./results_critic_iterations --critic_repeat $iter --name_prefix ITER_CRITIC >> results_critic_iterations/$1.log 2>&1
#    done
#done
