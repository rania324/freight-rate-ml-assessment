# Freight Rate Prediction

## Project Overview

This project predicts freight rates using historical load data.

The development dataset contains freight load information such as pickup location, delivery location, distance, equipment type, weight, market index, quote signal, and date.

## Approach

The following steps were used:

1. Loaded and explored the training data.
2. Cleaned invalid negative weight values.
3. Filled missing weight and market index values using the median.
4. Created date-based features:

   * year
   * month
   * day
   * day of week
5. Created a route feature from pickup and delivery locations.
6. Used a time-based train/validation split:

   * Training: January–September 2025
   * Validation: October 2025
7. Trained a CatBoost regression model with categorical features.
8. Used early stopping to select the best number of iterations.
9. Retrained the final model on the complete development dataset.
10. Generated predictions for the 12,000 validation loads.
11. Generated predictions for the December 2025 chart inputs.

## Model

CatBoostRegressor was selected because the dataset contains categorical features such as pickup, delivery, equipment, and route.

The final model used:

* Iterations: 208
* Depth: 8
* Learning rate: 0.05
* Loss function: RMSE
* Random seed: 42

## Validation Results

Using the time-based validation set:

* MAE: 115.08
* RMSE: 647.20
* R²: 0.821

## Output Files

* `validation_predictions.csv` — predictions for the 12,000 validation loads.
* `december-chart-inputs.csv` — December predictions.
* `scorer_results/candidate_december.png` — December prediction chart.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training and prediction script:

```bash
python train_model.py
```

Run the assessment scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

The scorer generates:

```text
scorer_results/candidate_december.png
```
