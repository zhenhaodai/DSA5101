#!/bin/bash
# MovieLens 20M Dataset Download Script

echo "=================================================="
echo "  MovieLens 20M Dataset Downloader"
echo "=================================================="
echo ""
echo "Downloading MovieLens 20M dataset..."
echo "Note: Dataset size is ~190MB, download may take a few minutes"
echo ""

# Create data directory
mkdir -p data
cd data

# Download dataset
echo "Downloading from GroupLens..."
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

# Extract
echo ""
echo "Extracting dataset..."
unzip -q ml-20m.zip

# Cleanup
echo "Cleaning up..."
rm ml-20m.zip

echo ""
echo "=================================================="
echo "  Download Complete!"
echo "=================================================="
echo "Dataset location: ./data/ml-20m/"
echo "Files:"
echo "  - ratings.csv (20M ratings)"
echo "  - movies.csv (27K movies)"
echo ""
