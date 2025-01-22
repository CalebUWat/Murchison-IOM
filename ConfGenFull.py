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
num_conformers = 100             # Number of conformers to generate
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


""" Butina Method to Cluster the Conformers based on RMSD """
# Load the RMSD matrix from the CSV file
rmsd_df = pd.read_csv("OUTPUT_rmsd_matrix.csv", index_col=0)

# Convert the RMSD DataFrame to a distance matrix (flattened)
rmsd_matrix = rmsd_df.values
num_molecules = rmsd_matrix.shape[0]

# Flatten the upper triangular part of the RMSD matrix (Butina requires a 1D distance array)
distances = []
for i in range(num_molecules):
    for j in range(i + 1, num_molecules):
        distances.append(rmsd_matrix[i, j])

# === STEP 1: Determine the RMSD Cut-off ===
# Ask the user for the RMSD cut-off
rmsd_cutoff = float(input("Enter the RMSD cut-off for clustering: "))

# === STEP 2: Perform Butina Clustering ===
# Perform clustering with the Butina algorithm
clusters = Butina.ClusterData(distances, num_molecules, rmsd_cutoff, isDistData=True)

# Print the number of clusters and their sizes
num_clusters = len(clusters)
print(f"\nMolecules have been clustered into {num_clusters} groups using RMSD cut-off = {rmsd_cutoff:.2f}")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {len(cluster)} molecules")

# === STEP 3: Save Clusters to Directories ===
clusters_dir = "./clusters"
os.makedirs(clusters_dir, exist_ok=True)

# Create a dictionary to track clusters
cluster_dict = {i + 1: cluster for i, cluster in enumerate(clusters)}

# Save files into corresponding cluster directories
for cluster_id, members in cluster_dict.items():
    cluster_dir = os.path.join(clusters_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    for member_idx in members:
        file_name = rmsd_df.index[member_idx]  # Get file name from the RMSD matrix index
        source_path = os.path.join(init_output_dir, file_name)
        target_path = os.path.join(cluster_dir, file_name)
        shutil.copy(source_path, target_path)

print(f"\nCluster files have been saved in the '{clusters_dir}' directory.")


""" Calculate Average Energy of each Cluster """
output_csv = 'cluster_average_energies.csv'  # Output CSV file to store average energies

# Dictionary to store the energy for each cluster
cluster_energies = {}

# Traverse through each cluster directory
for cluster in os.listdir(clusters_dir):
    cluster_path = os.path.join(clusters_dir, cluster)
    if os.path.isdir(cluster_path):
        cluster_energy_list = []
        # Traverse through each molecule file in the cluster directory
        for molecule_file in os.listdir(cluster_path):
            molecule_path = os.path.join(cluster_path, molecule_file)
            if molecule_file.endswith(('.mol', '.sdf', '.pdb')):  # Assuming supported formats
                energy = calculate_energy(molecule_path)
                if energy is not None:
                    cluster_energy_list.append(energy)

        # If there are energies in the cluster, calculate the average
        if cluster_energy_list:
            avg_energy = sum(cluster_energy_list) / len(cluster_energy_list)
            cluster_energies[cluster] = avg_energy
        else:
            print(f"No valid energies found for cluster {cluster}")

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(cluster_energies.items()), columns=['Cluster', 'Average Energy (kcal/mol)'])

# Save the results to a CSV file
df.to_csv(output_csv, index=False)

print(f"Average energies have been saved to {output_csv}")
