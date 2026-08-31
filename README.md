# SASRec Recommender System

A PyTorch-based sequential recommendation system that predicts the next item a user is likely to interact with, based on their historical sequence of actions.

## What Problem Does It Solve?

Traditional collaborative filtering models often ignore the order in which users interact with items. Sequential recommenders like this one treat user history as a time-ordered sequence, capturing short-term intent and long-term preferences to predict future interactions more accurately.

## System Pipeline & Architecture

1. **Data Processing:** Filters out sparse users and items (minimum 5 interactions), remaps IDs to contiguous integers, and chronologically sorts interactions.
2. **Dataset & Sampling:** Splits each user's timeline into `[train_sequence, validation_item, test_item]`. During training, it dynamically samples negative items to compute the loss.
3. **Model (SASRec):** A Transformer-based architecture using learned positional embeddings, multi-head self-attention with causality masking (so future items are not seen), and pointwise feed-forward networks with residual connections.
4. **Evaluation:** Evaluates validation and test sets by ranking the true target item against 100 sampled negative items, reporting Hit Rate (HR@10) and Normalized Discounted Cumulative Gain (NDCG@10).

## Important Results

The model was trained and evaluated on the **MovieLens-1M** dataset (6,040 users, 3,416 items, 999,611 interactions).

The best-performing configuration achieved the following metrics on the test set:
- **Test HR@10:** 0.7371
- **Test NDCG@10:** 0.4793
- **Final Training Loss:** 0.6075

### Ablation Study

We conducted a 2x2 experiment to understand the impact of maximum sequence length (`maxlen`) and dropout rate (`dropout`) on model performance. The combined changes yielded the best results.

| Configuration | Max Length | Dropout | Test HR@10 | Test NDCG@10 |
|---------------|------------|---------|------------|--------------|
| Baseline | 50 | 0.5 | 0.6781 | 0.4148 |
| Only dropout changed | 50 | 0.2 | 0.7154 | 0.4518 |
| Only maxlen changed | 200 | 0.5 | 0.6964 | 0.4384 |
| **Combined changes (Best)** | **200** | **0.2** | **0.7371** | **0.4793** |

*Note: You can view the full experimental results in `results/evaluation.json`.*

### Training Curve
During training on the best configuration (maxlen=200, dropout=0.2), the model smoothly converged over 200 epochs from an initial loss of ~5.6 down to ~0.6075.

## Installation and Execution

### Requirements
- Python 3.11+
- PyTorch
- NumPy
- tqdm
- PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the System

1. **Train the Model:**
```bash
python scripts/train.py --epochs 200 --maxlen 200 --dropout 0.2 --batch-size 128 --hidden-units 50 --num-blocks 2 --num-heads 1 --lr 0.001 --seed 42 --device cuda
```

2. **Evaluate a Checkpoint:**
```bash
python scripts/evaluate.py --checkpoint results/checkpoints/sasrec_movielens_maxlen200_dropout02.pt
```

3. **Generate Qualitative Recommendations:**
```bash
python scripts/qualitative_examples.py
```

## Expected Output

When running the qualitative examples script, the system outputs the recent history, held-out targets, and Top-5 recommended movies for selected users. For example:

```text
==========================================
User ID: 100
==========================================
Recent Watch History (last 10 items):
  - 2001: A Space Odyssey (1968) [Drama|Mystery|Sci-Fi|Thriller] (ID: 924)
  - Fargo (1996) [Crime|Drama|Thriller] (ID: 608)
  - GoodFellas (1990) [Crime|Drama] (ID: 1213)
  - Wizard of Oz, The (1939) [Adventure|Children's|Drama|Musical] (ID: 919)

Held-out targets:
  Validation (Next): Like Water for Chocolate (Como agua para chocolate) (1992) [Drama|Romance] (ID: 265)
  Test (After Valid): Apocalypse Now (1979) [Drama|War] (ID: 1208)

Top-5 Recommended Movies:
  1. Country (1984) [Drama] (ID: 3110)
  2. That's Life! (1986) [Drama] (ID: 3465)
  3. Simon Birch (1998) [Drama] (ID: 2236)
  4. Blow-Out (La Grande Bouffe) (1973) [Drama] (ID: 3655)
  5. Mo' Better Blues (1990) [Drama] (ID: 3425)
```

## Project Structure

```text
sasrec-recommender/
├── configs/
│   └── default.yaml                # Configuration defaults
├── data/
│   ├── raw/                        # Raw dataset files (e.g. ratings.dat, movies.dat)
│   └── processed/                  # Processed sequential dataset text files
├── results/
│   ├── checkpoints/                # Saved PyTorch models (*.pt)
│   ├── evaluation.json             # Aggregate metrics from experiments
│   └── qualitative_examples.txt    # Output from the qualitative recommendations script
├── scripts/
│   ├── train.py                    # Main training CLI
│   ├── evaluate.py                 # Checkpoint evaluation CLI
│   └── qualitative_examples.py     # Script to generate top-k examples
├── src/
│   ├── data/                       # Data processing, dataset, and negative sampler
│   ├── evaluation/                 # Metrics and evaluation logic
│   ├── models/                     # SASRec architecture, attention, layers
│   └── training/                   # PyTorch training loops
├── .gitignore                      # Git ignore file
├── README.md                       # This documentation
└── requirements.txt                # Python dependencies
```

## Reproducibility
All experiments were run with random seed `42`. To guarantee fully deterministic behavior across runs, ensure that PyTorch is configured for deterministic operations and the `--seed 42` flag is used on execution. 

## Limitations
- **Cold Start:** Users or items with fewer than 5 interactions are filtered out. The model does not natively handle brand new users or items without retraining.
- **Side Information:** This implementation purely uses item IDs and sequences. It does not integrate auxiliary data such as movie genres, timestamps of interaction, or user demographics.

## Future Improvements
- **Incorporating Item Features:** Feeding movie genres or descriptions into the item embeddings could improve representations, especially for items with fewer interactions.
- **Handling Long Sequences:** Exploring techniques like sparse attention or memory networks for sequences much larger than length 200.


