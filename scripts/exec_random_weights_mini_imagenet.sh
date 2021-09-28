#!/bin/bash

python3 training.py --model random_weights --dataset mini_imagenet

echo ""

python3 retraining.py

echo ""

python3 evaluation.py
