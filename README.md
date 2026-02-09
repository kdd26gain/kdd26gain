---

# GAIN: An Accurate Stock Price Prediction with Graph-based Capital Flow Inference Network

**[Under Review] Submitted to KDD 2026**

## Directory Structure

The project is organized as follows:

```bash
.
├── competitors/           # Competitive baseline models and runner script
│   ├── ALSTM.py           # Implementation of ALSTM
│   ├── DTML.py            # Implementation of DTML
│   ├── MASTER.py          # Implementation of MASTER
│   ├── THGNN.py           # Implementation of THGNN
│   ├── MGDPR.py           # Implementation of MGDPR
│   └── run_competitors.sh # Automated script for running all baselines
├── data/                  # Place raw CSV datasets here
├── model/                 # Trained model checkpoints are saved here
├── requirements.txt       # Python dependencies
├── src/                   # GAIN Source code
│   ├── main.py            # Main entry point for training and evaluation
│   ├── preprocess.py      # Data preprocessing pipeline
│   ├── engine.py          # Model architectures and metric calculations
│   ├── stgcn.py           # Core GAIN and ST-GCN layer implementations
│   ├── utils_path.py      # Path management utilities
│   ├── reproduce_experiment.py # Logic for reproducing paper results
│   └── reproduce.sh       # Full reproduction script (Preprocessing + Experiments)
└── README.md

```

## Requirements

* Python 3.8+
* PyTorch >= 1.10

To install the required dependencies:

```bash
pip install -r requirements.txt

```

## Data Preparation

Place your region-specific CSV files into the `data/` directory. The system expects files named in the following format:

* `stocks_{REGION}.csv` (e.g., `stocks_USA.csv`, `stocks_CHN.csv`, `stocks_EUR.csv`, `stocks_KOR.csv`)
* `macro_{REGION}.csv` (e.g., `macro_usa.csv`)
* `index_{REGION}.csv` (Required for index-based macro features)

## Usage

### 1. GAIN Reproduction (Main Model)

We provide a comprehensive shell script, `reproduce.sh`, to reproduce the results reported in the paper. This script automates the entire pipeline:

1. **Preprocessing**: Automatically generates tensor data for all regions (CHN, USA, EUR, KOR) with specific start dates and lag settings (10 and 20), applying the index option.
* **CHN**: From 2013-01-07
* **USA**: From 2013-01-03
* **EUR**: From 2013-01-03
* **KOR**: From 2018-01-01


2. **Experiment**: Runs the training and evaluation for **Best ASR** setting.
3. **Reporting**: Outputs the final metrics to `src/reproduce_report.txt`.

To run the reproduction suite:

```bash
cd src
chmod +x reproduce.sh
./reproduce.sh

```

### 2. Manual Training (GAIN)

If you wish to train a single model manually without the reproduction script:

**Step 1: Preprocess**

```bash
cd src
# Example for USA region, Lag 20
python main.py --preprocess --region USA --lags 20 --train_from 2013-01-03 --index

```

**Step 2: Train**

```bash
cd src
python main.py --region USA \
    --train_from 2013-01-03 --valid_from 2024-01-01 \
    --test_from 2025-01-01 --test_to 2025-03-31 \
    --epochs 200 --batch 32 --lr 0.001 \
    --index

```

**Key Arguments:**

* `--index`: Incorporates market index data into macro features.

### 3. Competitive Methods (Baselines)

We provide a dedicated script to reproduce the performance of competitive baselines (**ALSTM, DTML, MASTER, THGNN, MGDPR**) across all datasets.

The script automatically handles execution order and cache management to prevent data collision between standard and special experiment settings (e.g., CIKM22, IEEE24).

To run all competitors:

```bash
cd competitors
chmod +x run_competitors.sh
./run_competitors.sh

```

This will sequentially execute all baseline models for all regions (USA, CHN, EUR, KOR) and special settings, outputting the performance metrics for each.