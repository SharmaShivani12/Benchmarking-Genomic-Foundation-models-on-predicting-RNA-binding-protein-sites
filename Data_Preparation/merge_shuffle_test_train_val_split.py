import pandas as pd
import random
from sklearn.model_selection import train_test_split

def merge_and_shuffle(positive_file, negative_file, output_file):
    """
    Merges and shuffles positive and negative sequence files.

    Args:
    - positive_file (str): Path to the CSV file with positive sequences.
    - negative_file (str): Path to the CSV file with negative sequences.
    - output_file (str): Path to the output merged and shuffled CSV file.
    """
    # Load the positive and negative files into DataFrames
    positive_df = pd.read_csv(positive_file)
    negative_df = pd.read_csv(negative_file)
    
    # Merge the two DataFrames
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle the combined DataFrame
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the shuffled DataFrame to a file
    shuffled_df.to_csv(output_file, index=False)
    print(f"Merged and shuffled file saved to: {output_file}")



def split_dataset(merged_file, train_file, validation_file, test_file, train_ratio=0.6, val_ratio=0.2):
    """
    Splits a merged dataset into training, validation, and test sets using stratified sampling.

    Args:
    - merged_file (str): Path to the merged CSV file.
    - train_file (str): Path to save the training set.
    - validation_file (str): Path to save the validation set.
    - test_file (str): Path to save the test set.
    - train_ratio (float): Ratio of data to allocate for training (default: 0.6).
    - val_ratio (float): Ratio of data to allocate for validation (default: 0.2).
    """
    # Load the merged dataset
    df = pd.read_csv(merged_file)
    
    # Calculate the test size based on the remaining data after the training set
    remaining_ratio = 1 - train_ratio
    val_ratio_adjusted = val_ratio / remaining_ratio  # Adjust validation size for split after train set is removed

    # Split the dataset into train and temp (validation + test) with stratify
    train_data, temp_data = train_test_split(
        df,
        test_size=remaining_ratio,
        stratify=df['label'],  # 🚀 Stratify by the label column
        random_state=42
    )
    
    # Split temp into validation and test sets with stratify
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,  # Split equally
        stratify=temp_data['label'],  # 🚀 Stratify by the label column
        random_state=42
    )

    # Save the datasets to respective files
    train_data.to_csv(train_file, index=False)
    validation_data.to_csv(validation_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Dataset split completed:")
    print(f"- Training set: {len(train_data)} samples saved to {train_file}")
    print(f"- Validation set: {len(validation_data)} samples saved to {validation_file}")
    print(f"- Test set: {len(test_data)} samples saved to {test_file}")



# Main script
if __name__ == "__main__":
    # File paths
    positive_file = "path to your file"
    negative_file = "path to your file"
    merged_output_file = " path to your file"
    train_file = "path to your file"
    validation_file = "Path to your file"
    test_file = "Path to your file."

    # Merge and shuffle the datasets
    merge_and_shuffle(positive_file, negative_file, merged_output_file)
    
    # Split the merged dataset into training, validation, and test sets
    split_dataset(merged_output_file, train_file, validation_file, test_file)
