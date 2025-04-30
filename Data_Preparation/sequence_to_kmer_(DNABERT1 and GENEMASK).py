import pandas as pd

def generate_kmers(sequence, k):
    """
    Generate k-mers of size k from a sequence.
    """
    if len(sequence) < k:
        return []  # Return empty list if sequence is shorter than k
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Load CSV file
input_csv = ""  # Replace with your input CSV file name
output_tsv = ""  # Replace with your output TSV file name

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Ensure the sequence and label columns exist
sequence_column = "sequence"  # Column with sequences
label_column = "label"  # Column with labels

if sequence_column not in df.columns or label_column not in df.columns:
    raise ValueError(f"Columns '{sequence_column}' and/or '{label_column}' not found in the CSV file.")

# Generate k-mers and reformat the output for a TSV file
k = 6  # k-mer size
output_rows = []

for _, row in df.iterrows():
    sequence = row[sequence_column]
    label = row[label_column]
    kmers = generate_kmers(sequence, k)
    output_rows.append({
        sequence_column: " ".join(kmers),  # Join k-mers with a space
        label_column: label
    })

# Create a new DataFrame for the k-mers
output_df = pd.DataFrame(output_rows)

# Save the new DataFrame to a TSV file using to_tsv format
output_df.to_csv(output_tsv, sep="\t", index=False)

print(f"k-mers with labels have been generated and saved to {output_tsv}.")


