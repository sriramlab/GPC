split = 'afr'

filename = f'../../results/b38/{split}/data/{split}_train'
input_file = f"{filename}.hapt"
output_file = f"{filename}_PADDED.hapt"
target_features = 16383

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.strip().split()  # Split the line into parts
        metadata = parts[:2]  # Keep the first two columns
        features = parts[2:]  # The binary features
        
        # Pad with zeros if necessary
        padded_features = features + ["0"] * (target_features - len(features))
        
        # Write the new line
        outfile.write(" ".join(metadata + padded_features) + "\n")

print("Padding complete. Output saved to", output_file)