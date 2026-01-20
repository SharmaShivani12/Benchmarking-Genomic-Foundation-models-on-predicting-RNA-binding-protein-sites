import pandas as pd
import os

def generate_kmers(sequence, k=6):
    """Convert a DNA sequence into k-mers."""
    sequence = sequence.strip().upper().replace(" ", "")
    if len(sequence) < k:
        return ""  # Skip sequences shorter than k
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def convert_and_kmerize(input_file, k=6, output_file=None):
    # Read CSV
    df = pd.read_csv(input_file)

    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input file must contain 'sequence' and 'label' columns.")

    # Apply k-mer conversion
    df['sequence'] = df['sequence'].astype(str).apply(lambda x: generate_kmers(x, k))

    # Remove any rows with empty k-mers
    df = df[df['sequence'].str.strip() != ""]

    # Output filename
    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}_k{k}.tsv"

    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False)
    print(f"✅ Done. Converted file saved to: {output_file}")

if __name__ == "__main__":
    # 🔧 Hardcoded input and output paths
    input_file = "C:/Users/shiva/Documents/project_essentials/Datasets/HNRNAPL/validation_sequences.csv"  # <-- change this to your actual path
    output_file = "C:/Users/shiva/Documents/project_essentials/Datasets/HNRNAPL/DNB1/dev.tsv" # or set to "/path/to/output.tsv"
    k = 6  # You can change this to whatever k-mer size you want

    convert_and_kmerize(input_file, k=k, output_file=output_file)
