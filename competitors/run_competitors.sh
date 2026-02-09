#!/bin/bash

# ==============================================================================
# [run_competitors.sh]
# Competitor Experiment Replication Script
# Handles execution order and cache cleaning to prevent data collision.
# Models: ALSTM, DTML, MASTER, THGNN, MGDPR
# ==============================================================================

# List of models to run
MODELS=("ALSTM" "DTML" "MASTER" "THGNN" "MGDPR")

echo "========================================================"
echo "Starting Competitors Experiment Replication"
echo "Models: ${MODELS[*]}"
echo "========================================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "########################################################"
    echo "Running Experiments for Model: $model"
    echo "########################################################"

    # [Cache Directory Logic]
    # ALSTM uses 'cache_lstm', others use 'cache_{lowercase_model}'
    if [ "$model" == "ALSTM" ]; then
        CACHE_DIR="cache_lstm"
    else
        MODEL_LOWER=$(echo "$model" | tr '[:upper:]' '[:lower:]')
        CACHE_DIR="cache_${MODEL_LOWER}"
    fi

    echo "[Info] Cache Directory identified as: $CACHE_DIR"

    # ------------------------------------------------------------------
    # 1. Standard Datasets (USA, CHN, EUR, KOR)
    # ------------------------------------------------------------------

    # USA (Standard)
    echo "[$model] Running Standard USA (S&P500)..."
    python ${model}.py --region "S&P500" --train_from 2013-01-01 --valid_from 2024-01-01 --test_from 2024-10-01 --test_to 2025-03-31 --n_seeds 5

    # CHN (Standard)
    echo "[$model] Running Standard CHN (CSI300)..."
    python ${model}.py --region "CSI300" --train_from 2013-01-01 --valid_from 2024-01-01 --test_from 2024-10-01 --test_to 2025-03-31 --n_seeds 5

    # EUR (Standard)
    echo "[$model] Running Standard EUR (EURO50)..."
    python ${model}.py --region "EURO50" --train_from 2013-01-01 --valid_from 2024-01-01 --test_from 2024-10-01 --test_to 2025-03-31 --n_seeds 5

    # KOR (Standard)
    echo "[$model] Running Standard KOR (KP200)..."
    python ${model}.py --region "KP200" --train_from 2013-01-01 --valid_from 2024-01-01 --test_from 2024-10-01 --test_to 2025-03-31 --n_seeds 5


    # ------------------------------------------------------------------
    # 2. Special Settings (Cache Collision Handling)
    #    IEEE24 -> USA collision / CIKM22 -> CHN collision
    # ------------------------------------------------------------------

    # [IEEE24 Experiment]
    # Action: Delete existing USA cache files to force regeneration
    if [ -d "$CACHE_DIR" ]; then
        echo "[$model] Clearing USA cache for IEEE24 experiment..."
        rm -f ${CACHE_DIR}/*USA*
    fi

    echo "[$model] Running IEEE24 (S&P500)..."
    python ${model}.py --region "S&P500" --train_from 2013-01-01 --valid_from 2015-01-01 --test_from 2015-07-01 --test_to 2027-12-31 --n_seeds 5


    # [CIKM22 Experiment]
    # Action: Delete existing CHN cache files to force regeneration
    if [ -d "$CACHE_DIR" ]; then
        echo "[$model] Clearing CHN cache for CIKM22 experiment..."
        rm -f ${CACHE_DIR}/*CHN*
    fi

    echo "[$model] Running CIKM22 (CSI300)..."
    python ${model}.py --region "CSI300" --train_from 2016-01-01 --valid_from 2019-10-01 --test_from 2020-01-01 --test_to 2020-12-31 --n_seeds 5

done

echo "========================================================"
echo "All Experiments Completed Successfully."
echo "========================================================"