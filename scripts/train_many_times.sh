#!/bin/bash

python3 ./dataset_downloader.py -d omniglot

for i in {1..5}
do
    python3 training.py -m vanilla -d omniglot
    mv results/ "results-$i/"
done
