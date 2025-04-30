from Bio import SeqIO

def process_fasta(input_file, output_file, label):
    """
    Converts a FASTA file to the desired CSV format.
    
    Args:
    - input_file (str): Path to the input FASTA file.
    - output_file (str): Path to the output CSV file.
    - label (int): Numerical label to assign to sequences (1 for positive, 0 for negative).
    """
    with open(output_file, "w") as out_file:
        # Write the header row
        out_file.write("sequence,label\n")
        
        # Parse the FASTA file and write each sequence with the label
        for record in SeqIO.parse(input_file, "fasta"):
            sequence = str(record.seq)
            out_file.write(f"{sequence},{label}\n")

# Define file paths and labels
positive_fasta = "path to your file"
negative_fasta = "path to your file"
positive_output = "path to your file "
negative_output = "path to your file."

# Process the files
process_fasta(positive_fasta, positive_output, label=1)
process_fasta(negative_fasta, negative_output, label=0)

print(f"Files processed:\n- {positive_output}\n-{negative_output}\n")
