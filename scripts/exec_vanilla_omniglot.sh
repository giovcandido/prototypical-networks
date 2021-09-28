#!/bin/bash

python3 dataset_downloader.py --dataset omniglot

echo ""

python3 training.py --model vanilla --dataset omniglot

echo ""

python3 retraining.py

echo ""

python3 evaluation.py
