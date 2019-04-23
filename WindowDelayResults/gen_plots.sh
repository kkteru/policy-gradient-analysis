#!/bin/bash

# Script to run the delay experiment

python plot.py --path ~/Desktop/Hopper-v1/WindowExp/19-04-22_60_0/returns_eval.npy \
 ~/Desktop/Hopper-v1/WindowExp/19-04-21_120_0/returns_eval.npy \
 ~/Desktop/Hopper-v1/WindowExp/19-04-21_200_0/returns_eval.npy \
 ~/Desktop/Hopper-v1/WindowExp/19-04-21_300_0/returns_eval.npy \
 ~/Desktop/Hopper-v1/DelayExp/19-04-21_10000_0/returns_eval.npy \
 --label 60 120 200 300 Baseline --title HPWindowReward --xlabel Window #--no_std

python plot.py --path ~/Desktop/Hopper-v1/DelayExp/19-04-21_10000_0/returns_eval.npy \
 ~/Desktop/Hopper-v1/DelayExp/19-04-21_10000_30/returns_eval.npy \
 ~/Desktop/Hopper-v1/DelayExp/19-04-21_10000_50/returns_eval.npy \
 ~/Desktop/Hopper-v1/DelayExp/19-04-21_10000_70/returns_eval.npy \
 ~/Desktop/Hopper-v1/DelayExp/19-04-21_10000_100/returns_eval.npy \
 --label Baseline 30 50 70 100 --title HPDelayReward --xlabel Delay #--no_std

python plot.py --path ~/Desktop/InvertedDoublePendulum-v1/WindowExp/19-04-22_60_0/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/WindowExp/19-04-22_120_0/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/WindowExp/19-04-22_200_0/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/WindowExp/19-04-22_300_0/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/DelayExp/19-04-21_10000_0/returns_eval.npy \
 --label 60 120 200 300 Baseline --title IPWindowReward --xlabel Window #--no_std

python plot.py --path ~/Desktop/InvertedDoublePendulum-v1/DelayExp/19-04-21_10000_0/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/DelayExp/19-04-21_10000_30/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/DelayExp/19-04-21_10000_50/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/DelayExp/19-04-21_10000_70/returns_eval.npy \
 ~/Desktop/InvertedDoublePendulum-v1/DelayExp/19-04-21_10000_100/returns_eval.npy \
 --label Baseline 30 50 70 100 --title IPDelayReward --xlabel Delay #--no_std

python plot.py --path ~/Desktop/HalfCheetah-v1/WindowExp/19-04-03_60_0/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/WindowExp/19-04-03_120_0/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/WindowExp/19-04-03_200_0/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/WindowExp/19-04-03_300_0/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/DelayExp/19-04-03_10000_0/returns_eval.npy \
 --label 60 120 200 300 Baseline --title HCWindowReward --xlabel Window #--no_std

python plot.py --path ~/Desktop/HalfCheetah-v1/DelayExp/19-04-03_10000_0/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/DelayExp/19-04-03_10000_30/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/DelayExp/19-04-03_10000_50/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/DelayExp/19-04-03_10000_70/returns_eval.npy \
 ~/Desktop/HalfCheetah-v1/DelayExp/19-04-03_10000_100/returns_eval.npy \
 --label Baseline 30 50 70 100 --title HCDelayReward --xlabel Delay #--no_std