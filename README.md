# learntLoop: Support Vector Machine (SVM) Classifier for tRNA Sequence Data

## Overview
The learntLoop class is a tool designed for training and evaluating Support Vector Machine (SVM) classifiers on tRNA sequence data obtained from Nanopore reads. It accepts input in the form of training and testing arrays, typically provided as .tsv or .csv files. The class performs several tasks, including one-hot encoding on the input data, shuffling and concatenating the training data, training an SVM classifier, predicting labels for testing data, and analyzing the results.

## Features
One-Hot Encoding: Converts input data into a one-hot encoded format, enabling SVM training and prediction.
Data Shuffling: Randomly shuffles and concatenates training data to ensure balanced training samples.
Classification: Utilizes SVM classifier to predict labels for testing data.
Result Analysis: Provides detailed analysis including accuracy, confusion matrix, and classification report.
Individual Read Query: Optionally allows querying of individual reads for True Positives (TP), False Positives (FP), False Negatives (FN), or True Negatives (TN).

## Usage

```python learntLoop.py -T <training_files> -t <testing_files> [options]```

## Arguments
- -T, --Training: Input at least two training array files.
- -t, --testing: Input at least two testing array files.
- -l, --label: Designate the training label associated with the testing array.
- -q, --query-reads: Print out True Positive (TP), False Positive (FP), False Negative (FN), or True Negative (TN) classified reads.
- -s, --spike-in: Spike in a percentage of reads from another array.
- -r, --read-titrant: Input the array file containing reads that will be titrated.
- -tl, --titrant-label: Assign the titrated reads array a label.
- -c, --classification: Use One-Versus-Rest (ovr) or One-Versus-One (ovo) classification (Default ovr).

## Example

```python learntLoop-yeast.py -T ./IVT/training/Saccharomyces_cerevisiae_chrI_trna4-SerAGA.tsv ./WT/training/Saccharomyces_cerevisiae_chrI_trna4-SerAGA.tsv -t ./IVT/testing/Saccharomyces_cerevisiae_chrI_trna4-SerAGA.tsv ./WT/testing/Saccharomyces_cerevisiae_chrI_trna4-SerAGA.tsv```

## Requirements
- Python 3.x
- pandas
- scikit-learn
