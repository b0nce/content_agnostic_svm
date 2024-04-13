# Compressed File Kernel for SVM

This project provides an implementation of a custom kernel for Support Vector Machines that utilizes [normalized compression distance (or NCD)](https://en.wikipedia.org/wiki/Normalized_compression_distance). The kernel is designed to work with file paths. This method can be particularly useful in scenarios where file content similarity affects the outcome, such as in document classification or in distinguishing types of media files.

## System Requirements
Actual requirements are the following:
- numpy
- joblib
- zlib (usually included with Python)

But usually you'd also like to have something like:
- scikit-learn
- pandas

Latter two will have everything you need (like numpy and joblib), so I've included them to requirements.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/b0nce/content_agnostic_svm.git
cd content_agnostic_svm
```

Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage
To use this SVM kernel, you need to have a dataset where each instance corresponds to a file path pointing to the file to be used in the classification. Ensure your dataset includes a 'path' column with file paths and a 'label' column with the target labels.

```python
import pandas as pd
from sklearn.svm import SVC
from ncd_kernel import get_kernel

train_df = pd.read_csv("some_random_train_dataset.csv")
test_df = pd.read_csv("some_random_test_dataset.csv")

assert "path" in train_df and "path" in test_df  # X
assert "label" in train_df and "label" in test_df  # y

kernel = get_kernel(train_df["path"].values, n_jobs=-1)
clf = SVC(kernel=kernel)

clf.fit(train_df["path"].values, train_df["label"].values)

test_accuracy = clf.score(test_df["path"].values, test_df["label"].values)

print(f"{test_accuracy = }")
```


