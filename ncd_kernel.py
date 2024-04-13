import zlib
import numpy as np
from joblib import Parallel, delayed


def compressed_size(file_path, file_path_2=None):
    """Returns the compressed size of one or two files combined."""
    with open(file_path, "rb") as file:
        content = file.read()
    if file_path_2:
        with open(file_path_2, "rb") as file:
            file2_content = file.read()
            # Resolving K(a, b) = K(b, a)
            if content > file2_content:
                content += file2_content
            else:
                content = file2_content + content
    compressed_content = zlib.compress(content)
    return len(compressed_content)


def precompute_single_sizes(X, n_jobs=-1):
    """Precomputes and returns a dictionary of compressed sizes for a list of file paths."""
    def compute_single_compressed_size(fpath):
        return fpath, compressed_size(fpath)

    unique_files = list(set(X))
    compressed_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_compressed_size)(fpath) for fpath in unique_files
    )
    return dict(compressed_results)


def get_kernel(X, reg=2, n_jobs=-1):
    """Generates a kernel function based on compressed file sizes and a regularization term."""
    compressed_single = precompute_single_sizes(X, n_jobs=n_jobs)

    def kernel(X, Y):
        # Precompute any missing file sizes from Y
        new_singles = set(X).union(set(Y)) - set(compressed_single.keys())
        new_singles = precompute_single_sizes(list(new_singles), n_jobs=n_jobs)
        compressed_single.update(new_singles)
        
        # Generate all combinations of file indices and paths
        combinations = [(i, fpath1, j, fpath2) for i, fpath1 in enumerate(X) for j, fpath2 in enumerate(Y)]

        # Calculate the normalized distance for each pair
        def compute_element(i, fpath1, j, fpath2):
            combined_size = compressed_size(fpath1, fpath2)
            norm_distance = combined_size - min(compressed_single[fpath1], compressed_single[fpath2])
            norm_distance /= max(compressed_single[fpath1], compressed_single[fpath2])
            return (i, j, reg - norm_distance)

        # Calculate the matrix in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_element)(i, fpath1, j, fpath2)
            for i, fpath1, j, fpath2 in combinations
        )

        # Populate the results matrix
        res = np.zeros((len(X), len(Y)))
        for i, j, value in results:
            res[i, j] = value

        return res

    return kernel


if __name__ == "__main__":
    # Example of usage
    import pandas as pd
    from sklearn.svm import SVC

    train_df = pd.read_csv("some_random_train_dataset.csv")
    assert "path" in train_df  # X
    assert "label" in train_df  # y

    test_df = pd.read_csv("some_random_test_dataset.csv")
    assert "path" in test_df  # X
    assert "label" in test_df  # y

    ker = get_kernel(train_df["path"].values, n_jobs=-1)
    clf = SVC(kernel=ker)
    clf.fit(train_df["path"].values, train_df["label"].values)

    test_accuracy = clf.score(test_df["path"].values, test_df["label"].values)

    print(f"{test_accuracy = }")
