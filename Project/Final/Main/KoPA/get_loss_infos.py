def extract_loss_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip().startswith("{'loss':"):
                outfile.write(line)

# Example usage
input_filename = 'log.txt'
output_filename = 'lora-Llama-2-7b-alpaca-cleaned/training_log.txt'
extract_loss_lines(input_filename, output_filename)

print(f"Extracted lines written to {output_filename}")


