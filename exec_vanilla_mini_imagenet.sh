#!/bin/bash

python3 dataset_downloader.py --dataset mini_imagenet

echo ""

python3 training.py --model vanilla --dataset mini_imagenet

echo ""

python3 retraining.py

echo ""

python3 evaluation.py
