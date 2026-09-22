# Individual Task 2 - COSC2669 Case Studies in Data Science

Bias, fairness and learning-curve analysis for Individual Task 2, Part 2.
Shivkumar Mundhe (s4178111), RMIT University.

This repository holds the analysis script and outputs behind Part 2 of the
report. It extends Pipeline 2 from Individual Task 1 - a Random Forest
predicting corporate bankruptcy from the Taiwanese Bankruptcy Prediction
dataset - with three additions:

1. a reproduction check and a cross-validation diagnostic
2. a learning curve across varying training-set sizes
3. a Fairlearn fairness audit using firm scale as a proxy sensitive attribute

Task 1 pipelines are in a separate repository:
https://github.com/databyshiv/asx-supervision-pipeline-project

## Contents

| File | Description |
|------|-------------|
| `task2_analysis.py` | The full analysis script |
| `outputs_task2/results.json` | Every number reported in Part 2 |
| `outputs_task2/fig_learning_curve.png` | Figure 1 in the report |
| `outputs_task2/fig_fairness_by_scale.png` | Figure 2 in the report |
| `requirements.txt` | Package versions used |

## Data

Download the Taiwanese Bankruptcy
Prediction collection from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction

Extract `data.csv`, rename it `taiwan_bankruptcy.csv`, and place it beside
`task2_analysis.py`. It has 6,819 rows and 96 columns, with a bankruptcy
rate of 3.23%.

## Running it

```bash
pip install -r requirements.txt
python task2_analysis.py
```

Results and figures are written to `outputs_task2/`.

## Version note

Results were produced with **scikit-learn 1.9.1** and **Fairlearn 0.14.0**
on macOS (Apple Silicon). The version matters. On scikit-learn 1.8.0 the
same script and seed give 13 true positives instead of 25, because the
predicted probabilities for borderline cases sit very close to the 0.5
decision threshold — only 20 of the 1,705 test cases fall between 0.45 and
0.55, so small numerical differences flip a number of them. Use the pinned
versions in `requirements.txt` to reproduce the figures in the report.

## What the script reports

- **Reproduction check** -  re-runs the Task 1 configuration and compares
  against the metrics reported there
- **Cross-validation diagnostic** - compares the unshuffled `cv=5` used in
  Task 1 against a shuffled `StratifiedKFold`, and computes the lag-1
