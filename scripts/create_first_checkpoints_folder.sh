#!/bin/bash

# Set base directory
BASE_DIR="../cl_models"

# Function to create seed subdirectories
create_seeds() {
    for seed in {0..4}; do
        mkdir -p "$1/seed_$seed"
    done
}

# CIFAR100
for split in 5_class 10_class 40_class 50_class; do
    DIR="$BASE_DIR/cifar100/$split"
    create_seeds "$DIR"
done

# Tiny-ImageNet
for split in 10_class 20_class 100_class; do
    DIR="$BASE_DIR/tiny-imagenet/$split"
    create_seeds "$DIR"
done

# ImageNet-Subset
for split in 5_class 10_class 40_class 50_class; do
    DIR="$BASE_DIR/imagenet-subset/$split"
    create_seeds "$DIR"
done

# ImageNet-1k
for split in 50_class 100_class 400_class 500_class; do
    DIR="$BASE_DIR/imagenet-1k/$split"
    create_seeds "$DIR"
done

# DomainNet
DIR="$BASE_DIR/domainnet/100_class"
create_seeds "$DIR"

echo "Full directory structure with seeds created under $BASE_DIR"
