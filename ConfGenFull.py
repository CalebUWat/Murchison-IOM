import os
import subprocess
import pandas as pd
import numpy as np
import shutil
import pymol
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
from rdkit.ML.Cluster import Butina

# Parameters
sdf_file = "murchison.sdf"          # Input SDF file
num_conformers = 1000             # Number of conformers to generate
init_output_dir = "conformers_output"  # Directory to save conformer files
init_energy_csv = "total_conformer_energies.csv" # Filename of csv with all generated conformers calculated energies


""" Generate Conformers """
def generate_conformers_to_files(sdf_file, num_conformers, init_output_dir):

    # Load molecules from the SDF file
    supplier = Chem.SDMolSupplier(sdf_file)

    # Extract the first molecule
    mol = supplier[0]
    if mol is None:
        print(f"Failed to load the first molecule from '{sdf_file}'. Please check the file.")
        return

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Set up the ETKDGv3 method parameters
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # Use all available CPU threads
    params.maxAttempts = 1000 # Allow more attempts to embed conformers
    params.pruneRmsThresh = -1.0  # RMSD threshold for pruning similar conformers
    params.randomSeed = 42 # Ensure reproducibility
    params.useRandomCoords = True # Generate random starting coordinates
    params.maxIterations = 1000  # Maximum iterations for embedding

    # Generate the conformers
    conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    num_generated = len(conformer_ids)
    print(f"Generated {num_generated} conformers for the first molecule in '{sdf_file}'.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Optimize each conformer and save to a separate SDF file
    for conf_id in range(num_generated):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        conformer_filename = os.path.join(init_output_dir, f"molecule_conformer_{conf_id + 1}.sdf")

        # Write the conformer to a file
        with SDWriter(conformer_filename) as writer:
            writer.write(mol, confId=conf_id)
        print(f"Conformer {conf_id + 1} saved to '{conformer_filename}'")


""" Calculate Energy for each Conformer into CSV """
def calculate_energy(molecule_file):

    try:
        # Run the obenergy command with MMFF94 force field
        result = subprocess.run(
            ['obenergy', '-ff', 'MMFF94', molecule_file],
            capture_output=True, text=True, check=True
        )

        # Look for the energy in the output (line starting with 'TOTAL ENERGY =')
        for line in result.stdout.splitlines():
            if "TOTAL ENERGY =" in line:
                # Extract the energy value (after the '=' and before 'kcal/mol')
                energy = line.split('=')[1].strip().split()[0]
                return float(energy)  # Return as float for saving

        # If no energy value found, return None
        print(f"Energy not found in the output for {molecule_file}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error calculating energy for {molecule_file}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for {molecule_file}: {e}")
        return None

def calculate_energies_in_directory(directory, output_csv):

    results = []  # List to store results

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Only process valid molecule files (.mol, .sdf, etc.)
        if filename.endswith(('.mol', '.sdf', '.pdb')):
            print(f"Processing {filename}...")
            energy = calculate_energy(file_path)

            # Append result as a tuple (filename, energy)
            results.append((filename, energy))
        else:
            print(f"Skipping non-molecule file: {filename}")

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results, columns=['Filename', 'Energy (kcal/mol)'])
    df.to_csv(output_csv, index=False)
    print(f"Energies saved to {output_csv}")

# Execute ConfGen and EnergyCalc
generate_conformers_to_files(sdf_file, num_conformers, init_output_dir)
calculate_energies_in_directory(init_output_dir, init_energy_csv)


""" Calculate RMSD Matrix for Conformers """
# Check if RMSD matrix already exists
rmsd_matrix_file = "OUTPUT_rmsd_matrix.csv"

if os.path.exists(rmsd_matrix_file):
    print(f"\n'{rmsd_matrix_file}' already exists. Skipping RMSD calculation.\n")
else:
    print(f"\n'{rmsd_matrix_file}' not found. Calculating RMSD matrix...\n")

    # Directory containing the molecular files
    molecule_files = [f for f in os.listdir(init_output_dir) if f.endswith((".pdb", ".mol", ".sdf"))]
    num_files = len(molecule_files)

    # Create an empty RMSD matrix (num_files x num_files)
    rmsd_matrix = np.zeros((num_files, num_files))

    # Initialize PyMOL without GUI
    pymol.finish_launching(['pymol', '-qc'])

    # Loop over each pair of molecules to calculate RMSD
    for i, ref_file in enumerate(molecule_files):
        ref_path = os.path.join(init_output_dir, ref_file)
        ref_name = os.path.splitext(ref_file)[0]

        # Load the reference structure into PyMOL
        cmd.load(ref_path, "reference")

        for j, target_file in enumerate(molecule_files):
            if i == j:
                rmsd_matrix[i, j] = 0.0  # RMSD of a molecule with itself is 0
                continue

            target_path = os.path.join(init_output_dir, target_file)
            target_name = os.path.splitext(target_file)[0]

            # Load the target structure
            cmd.load(target_path, target_name)

            # Perform the fit command and retrieve the RMSD value
            try:
                rmsd = cmd.fit(target_name, "reference")
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd  # Symmetric matrix
                print(f"Reference: {ref_file}, Target: {target_file}, RMSD = {rmsd:.2f}")
            except Exception as e:
                print(f"Error fitting {target_file} with reference {ref_file}: {e}")
                rmsd_matrix[i, j] = np.nan  # Handle failed fits

            # Remove the target structure from PyMOL
            cmd.delete(target_name)

        # Remove the reference structure from PyMOL
        cmd.delete("reference")

    # Quit PyMOL
    cmd.quit()

    # Save the RMSD matrix to a CSV file for inspection
    rmsd_df = pd.DataFrame(rmsd_matrix, index=molecule_files, columns=molecule_files)
    rmsd_df.to_csv(rmsd_matrix_file)
    print(f"\nRMSD matrix saved to '{rmsd_matrix_file}'.\n")
