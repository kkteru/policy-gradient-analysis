#!/bin/bash

# Script to run the delay experiment

python main_baseline.py --use_logger True --gpu=0

python main_baseline.py --use_logger True --delay 10 --gpu=0

python main_baseline.py --use_logger True --delay 20 --gpu=0

python main_baseline.py --use_logger True --delay 30 --gpu=0

python main_baseline.py --use_logger True --delay 40 --gpu=0

python main_baseline.py --use_logger True --delay 50 --gpu=0

python main_baseline.py --use_logger True --delay 60 --gpu=0

python main_baseline.py --use_logger True --delay 70 --gpu=0

python main_baseline.py --use_logger True --delay 80 --gpu=0

python main_baseline.py --use_logger True --delay 90 --gpu=0

python main_baseline.py --use_logger True --delay 100 --gpu=0

