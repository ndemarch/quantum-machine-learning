import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def match_and_remove_duplicates(d1_path, d2_path, col1, col2):
    """
    Matches two datasets based on specified columns, removes duplicates, and filters based on a condition.

    Parameters:
    d1_path (str): Path to the first dataset.
    d2_path (str): Path to the second dataset.
    col1 (str): Column in the first dataset for matching.
    col2 (str): Column in the second dataset for matching.

    Returns:
    tuple: A tuple containing the filtered and matched dataframes from both datasets.
    """
    # read the datasets
    d1 = pd.read_csv(d1_path)
    d2 = pd.read_csv(d2_path)
    # filter d1 based on d2's IDs
    d1_filtered = d1[d1[col1].isin(d2[col2])]
    # remove duplicates from d1
    d1_filtered_no_duplicates = d1_filtered.drop_duplicates(subset=[col1])
    # keep only galaxies where Error = 0
    d1_filtered_final = d1_filtered_no_duplicates[d1_filtered_no_duplicates["Error"]==0]
    # filter d2 based on d1's IDs
    d2_filtered = d2[d2[col2].isin(d1_filtered_final[col1])]
    # remove duplicates from d2
    d2_filtered_final = d2_filtered.drop_duplicates(subset=[col2])
    # assertion of final shapes
    assert d1_filtered_final.shape[0] == d2_filtered_final.shape[0]
    
    return d1_filtered_final, d2_filtered_final

def filter_columns(d1, d2):
    """
    Filters columns from the input dataframes.

    Parameters:
    d1 (DataFrame): First dataframe.
    d2 (DataFrame): Second dataframe.

    Returns:
    tuple: A tuple containing the filtered dataframes from both datasets.
    """     
    # columns to keep in d1
    d1_columns_to_keep = [
        'dr7objid', 
        'K', 
        'C', 
        'A', 
        'S', 
        'G2', 
        'H',
    ]
    # columns to keep in d2
    d2_columns_to_keep = [
        "OBJID",
        "SPIRAL",
        "ELLIPTICAL",
        "UNCERTAIN",
    ]
    
    return d1[d1_columns_to_keep], d2[d2_columns_to_keep]

def merge_and_filter_uncertain(d1, d2):
    """
    Merges two dataframes and filters rows based on a condition.

    Parameters:
    d1 (DataFrame): First dataframe.
    d2 (DataFrame): Second dataframe.

    Returns:
    DataFrame: Merged and filtered dataframe.
    """     
    # merge d1 and d2 on common ID's
    merged_df = pd.merge(d1, d2, left_on='dr7objid', right_on='OBJID', how='inner')
    merged_df = merged_df.drop(columns = "OBJID")
    # remove where we are uncertain of the morphology type
    merged_df = merged_df[merged_df["UNCERTAIN"] == 0]
    merged_df = merged_df.drop(columns = "UNCERTAIN")
    # statistical conditions
    merged_df = merged_df[(merged_df["G2"] > 0) & (merged_df["A"] > 0) & (merged_df["C"] > 0) & (merged_df["S"] > 0)]
    # reset index
    merged_df = merged_df.reset_index(drop=True)
    
    return merged_df

def create_binary_col(df):
    """
    Creates a binary column based on specified conditions.

    Parameters:
    df (DataFrame): Input dataframe.

    Returns:
    DataFrame: DataFrame with the binary column.
    """     
    if df["SPIRAL"].sum() + df["ELLIPTICAL"].sum() != df.shape[0]:
        raise ValueError("We have more than one defined class for object")
    # creating binary column based on spiral or not (i.e. elliptical)
    df["morphological_type"] = df["SPIRAL"]
    # assert
    sum = 0
    for i,j,k in zip(df["SPIRAL"],df["ELLIPTICAL"],df["morphological_type"]):
        if (i != j) and (k == i or k == j):
            sum += 1
    assert sum == df.shape[0]
    df = df.drop(columns = ["SPIRAL", "ELLIPTICAL"])
    
    return df

if __name__ in "__main__":
    # run our preprocessing steps
    d1_path = "./data/Barchi19_Morphology_Catalog.csv"
    d2_path = "./data/GalaxyZoo1_DR_table2.csv"
    col1, col2 = "dr7objid", "OBJID"

    d1_filtered, d2_filtered = match_and_remove_duplicates(d1_path, d2_path, col1, col2)
    d1_final, d2_final = filter_columns(d1_filtered, d2_filtered)
    merged_df = merge_and_filter_uncertain(d1_final, d2_final)
    df = create_binary_col(merged_df)

    df.to_csv("./data/processed_morphology_dataset.csv", index=False)