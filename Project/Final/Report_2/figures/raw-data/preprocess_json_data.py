import json

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Read the file containing the data
    input_file = "data.txt"

    # Read and parse the data
    data = []
    with open(input_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.replace("'", '"'))  # Convert single quotes to double quotes
                # Handle NaN values safely
                for key, value in entry.items():
                    if isinstance(value, str) and value.lower() == "nan":
                        entry[key] = np.nan  # Convert to proper NaN
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {line.strip()} - Error: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Remove duplicate epochs (keeping the first occurrence)
    df = df.drop_duplicates(subset=['epoch'], keep='first')

    # Save to a .dat file for pgfplots
    df.to_csv("lora-Llama-2-7b-alpaca-cleaned-finetune.dat", sep=" ", index=False, header=True)
    print("Filtered data saved to data.dat")
