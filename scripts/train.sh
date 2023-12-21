#!/bin/sh
accelerate launch --num_processes=4 src/train.py
