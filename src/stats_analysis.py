import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def stratified_subsample(df, class_label, total_samples, random_state=None):
    """
    Subsamples a dataframe in a stratified manner based on a specified class label.

    Parameters:
    df (DataFrame): The input dataframe.
    class_label (str): The name of the column containing the class labels.
    total_samples (int): The total number of samples to be subsampled.

    Returns:
    DataFrame: The subsampled dataframe.
    """
    if random_state is not None:
        np.random.seed(random_state)
    class_0_count = (df[class_label] == 0).sum()
    class_1_count = (df[class_label] == 1).sum()

    if class_0_count == 0 or class_1_count == 0:
        raise ValueError("One of the class counts is zero.")

    class_1_ratio = class_1_count / (class_0_count + class_1_count)

    class_1_samples = int(total_samples * class_1_ratio)
    class_0_samples = total_samples - class_1_samples

    class_0_data = df[df[class_label] == 0]
    class_1_data = df[df[class_label] == 1]

    class_0_sampled = class_0_data.sample(n=class_0_samples, replace=True)
    class_1_sampled = class_1_data.sample(n=class_1_samples, replace=True)

    subsampled_df = pd.concat([class_0_sampled, class_1_sampled])
    final_subsampled_df = subsampled_df.sample(frac=1).reset_index(drop=True)

    return final_subsampled_df

def distributions_and_correlations(X, name=''):
    """
    Plots feature distributions and correlation matrix.

    Parameters:
    X (DataFrame): Input features.
    name (str): Name to append to the output filenames.

    Returns:
    None
    """
    # histogram plot
    X.hist(bins=20, figsize=(9, 7))
    plt.tight_layout()
    plt.savefig("./plots/distributions" + str(name) + ".pdf", dpi=1200)
    plt.close()
    # correlation plot
    corr = X.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr, annot=True, cmap='PuBuGn', robust=True, mask=mask, vmin=-1, vmax=1)
    plt.savefig("./plots/correlations" + str(name) + ".pdf", dpi=1200)
    plt.close()

def feature_engineer(X):
    """
    Performs feature engineering by dropping highly correlated features.

    Parameters:
    X (DataFrame): Input features.

    Returns:
    DataFrame: Feature-engineered dataframe.
    """
    X.drop(columns=['G2', 'S'], inplace=True)
    return X

def standardize(X):
    """
    Standardizes input features using StandardScaler.

    Parameters:
    X (DataFrame): Input features.

    Returns:
    DataFrame: Standardized features dataframe.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)

    return pd.DataFrame(X_standardized, columns=X.columns)


if __name__ in "__main__":
    # load processed data
    df = pd.read_csv("../data/processed_morphology_dataset.csv")
    X_org = df.drop(columns = ["morphological_type","dr7objid"])
    X_org = standardize(X_org)
    distributions_and_correlations(X_org, name = "_all_features")
    print(f"Original data shape: {X_org.shape}")
    subsampled_df = stratified_subsample(df, "morphological_type", 200000, random_state=42)
    # split features and label
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type","dr7objid"])
    X = standardize(X)
    # save data for reproducability
    X.to_csv("../data/subsampled_data_features.csv", index=False)
    y.to_csv("../data/subsampled_data_labels.csv", index=False)
    # get distributions
    # feature engineer
    X_new = feature_engineer(X)
    X_new = standardize(X_new)
    #distributions_and_correlations(X_new, name = "_model2")
    # save feature engineered data
    X_new.to_csv("../data/subsampled_engineered_features.csv", index=False)
