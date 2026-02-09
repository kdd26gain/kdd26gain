#!/bin/bash

# reproduce.sh
# ------------------------------------------------------------------
# Step 1: Preprocessing (Data Generation)
# Step 2: Experiment Reproduction (Train/Eval)
# ------------------------------------------------------------------

# Stop execution if any command fails
set -e

echo "============================================================"
echo "Step 1: Running Preprocessing Pipeline..."
echo "============================================================"

# Function to run preprocessing for both lag 10 and 20
run_prep() {
    REGION=$1
    DATE=$2

    echo ">> Preprocessing Region: $REGION (Start: $DATE, Index: True)"

    # Lag 10
    echo "   - Generating Lag 10..."
    python main.py --preprocess --region $REGION --macro $REGION --lags 10 --train_from $DATE --index

    # Lag 20
    echo "   - Generating Lag 20..."
    python main.py --preprocess --region $REGION --macro $REGION --lags 20 --train_from $DATE --index
}

# Execute for each region with specific start dates
run_prep "CHN" "2013-01-07"
run_prep "USA" "2013-01-03"
run_prep "EUR" "2013-01-03"
run_prep "KOR" "2018-01-01"

echo ""
echo "============================================================"
echo "Step 2: Running Experiment Reproduction..."
echo "============================================================"

# Run the python script that handles Best MSE/ASR reproduction
python reproduce_experiment.py

echo ""
echo "✅ All tasks completed successfully."