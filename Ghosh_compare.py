import os
import pandas as pd
import numpy as np
import pymol
from pymol import cmd

# Configuration
input_dir = "7A_clusters"  # Directory containing molecule files
reference_files = ["ghosh_1.sdf", "ghosh_2.sdf", "ghosh_3.sdf"]  # List of 3 reference files
output_file = "RMSD_comparison.xlsx"  # Output Excel file

# Validate reference files
for ref_file in reference_files:
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference file '{ref_file}' not found.")

# Get all target molecule files from the directory
target_files = [f for f in os.listdir(input_dir) if f.endswith((".pdb", ".mol", ".sdf"))]
if not target_files:
    raise ValueError(f"No molecule files found in '{input_dir}'.")

# Initialize PyMOL without GUI
pymol.finish_launching(['pymol', '-qc'])

# Initialize a DataFrame for storing results
rmsd_results = pd.DataFrame(index=reference_files, columns=target_files)

# Calculate RMSD for each reference molecule against all target molecules
for ref_file in reference_files:
    ref_name = os.path.splitext(os.path.basename(ref_file))[0]
    cmd.load(ref_file, "reference")
    print(f"Loaded reference molecule: {ref_file}")

    # Remove hydrogen atoms from the reference molecule
    cmd.remove("hydro")

    for target_file in target_files:
        target_path = os.path.join(input_dir, target_file)
        target_name = os.path.splitext(target_file)[0]

        # Load the target structure
        cmd.load(target_path, target_name)

        # Remove hydrogen atoms from the target molecule
        cmd.remove(f"{target_name} and hydro")

        try:
            # Calculate RMSD
            rmsd = cmd.fit(target_name, "reference", matchmaker=4)
            rmsd_results.loc[ref_file, target_file] = rmsd
            print(f"RMSD between {ref_file} and {target_file} (without hydrogens): {rmsd:.2f}")
        except Exception as e:
            rmsd_results.loc[ref_file, target_file] = np.nan
            print(f"Error calculating RMSD for {ref_file} and {target_file}: {e}")

        # Remove the target molecule
        cmd.delete(target_name)

    # Remove the reference molecule
    cmd.delete("reference")

# Quit PyMOL
cmd.quit()

# Save the results to an Excel file
rmsd_results.to_excel(output_file)
print(f"\nRMSD comparison (without hydrogens) saved to '{output_file}'.")
