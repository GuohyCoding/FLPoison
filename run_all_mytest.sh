#!/bin/bash
# Run MyTest for all 6 datasets with specified epoch counts.
# Usage: bash run_all_mytest.sh [GPU_IDX]
# Example: bash run_all_mytest.sh 0

set -e
PYTHON=/home/hongyi/ENTER/envs/FLPoison/bin/python
GPU=${1:-0}

cd "$(dirname "$0")"

echo "=== Running MyTest on GPU $GPU ==="

$PYTHON run_mytest.py -datasets MNIST        -alg FedSGD -e 400  -gidx $GPU --force
$PYTHON run_mytest.py -datasets FashionMNIST -alg FedSGD -e 800  -gidx $GPU --force
$PYTHON run_mytest.py -datasets CHMNIST      -alg FedSGD -e 1800 -gidx $GPU --force
$PYTHON run_mytest.py -datasets CINIC10      -alg FedSGD -e 1700 -gidx $GPU --force
$PYTHON run_mytest.py -datasets CIFAR10      -alg FedSGD -e 1100 -gidx $GPU --force
$PYTHON run_mytest.py -datasets CIFAR100     -alg FedSGD -e 1700 -gidx $GPU --force

echo "=== All done. Generating convergence figures ==="
$PYTHON plot_convergence_figures.py
