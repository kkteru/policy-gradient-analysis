#!/bin/bash

# Script to run the delay experiment

python main_baseline.py \
--max_timesteps 1e3

python main_baseline.py \
--max_timesteps 1e3 \
--delay 10

python main_baseline.py \
--max_timesteps 1e3 \
--delay 20

python main_baseline.py \
--max_timesteps 1e3 \
--delay 30

python main_baseline.py \
--max_timesteps 1e3 \
--delay 40

python main_baseline.py \
--max_timesteps 1e3 \
--delay 50

python main_baseline.py \
--max_timesteps 1e3 \
--delay 60

python main_baseline.py \
--max_timesteps 1e3 \
--delay 70

python main_baseline.py \
--max_timesteps 1e3 \
--delay 80

python main_baseline.py \
--max_timesteps 1e3 \
--delay 90

python main_baseline.py \
--max_timesteps 1e3 \
--delay 100
