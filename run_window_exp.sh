#!/bin/bash

# Script to run the delay experiment

python main_baseline.py --use_logger True --window 10 --gpu=1

python main_baseline.py --use_logger True --window 20 --gpu=1

python main_baseline.py --use_logger True --window 40 --gpu=1

python main_baseline.py --use_logger True --window 60 --gpu=1

python main_baseline.py --use_logger True --window 80 --gpu=1

python main_baseline.py --use_logger True --window 100 --gpu=1

python main_baseline.py --use_logger True --window 120 --gpu=1

python main_baseline.py --use_logger True --window 140 --gpu=1

python main_baseline.py --use_logger True --window 160 --gpu=1

python main_baseline.py --use_logger True --window 180 --gpu=1

python main_baseline.py --use_logger True --window 200 --gpu=1

