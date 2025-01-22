import pandas as pd
import os
import shutil
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load the RMSD matrix from the CSV file
rmsd_df = pd.read_csv("OUTPUT_rmsd_matrix.csv", index_col=0)

# Convert the RMSD DataFrame to a distance matrix
rmsd_matrix = rmsd_df.values

# === STEP 1: Perform Hierarchical Clustering ===
# Perform hierarchical clustering
linked = sch.linkage(rmsd_matrix, method='complete')

# === STEP 2: Automatic Cluster Selection Based on Linkage ===
# Print the linkage matrix for debugging
print("\nLinkage matrix:")
print(linked)

# === STEP 3: Determine Optimal RMSD Cut-off ===
# Dynamically select a reasonable cut-off by examining the dendrogram visually
rmsd_cutoff = float(input("Enter the RMSD cut-off based on the dendrogram (look at the plot and select a reasonable value): "))

# === STEP 4: Form Clusters Based on Cut-off ===
from scipy.cluster.hierarchy import fcluster
labels = fcluster(linked, t=rmsd_cutoff, criterion='distance')

# Get the number of clusters generated
num_clusters = len(set(labels))
print(f"\nMolecules have been clustered into {num_clusters} groups using RMSD cut-off = {rmsd_cutoff:.2f}")

# === STEP 5: Save Clusters to Directories ===
output_dir = "./clusters"
os.makedirs(output_dir, exist_ok=True)

# Copy molecule files to their respective clusters
input_dir = "./conformers_output"  # Directory containing the molecule files

# Create a dictionary to track clusters
cluster_dict = {}

for i, file in enumerate(rmsd_df.index):
    cluster_id = labels[i]
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    # Copy or move the molecule file to the cluster directory
    source_path = os.path.join(input_dir, file)
    target_path = os.path.join(cluster_dir, file)
    shutil.copy(source_path, target_path)
    
    # Track the cluster and the file
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(file)

print(f"\nCluster files have been saved in the '{output_dir}' directory.")

