import pandas as pd
import numpy as np
import os
import shutil
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

# === Step 1: Load the RMSD Matrix ===
# Load the RMSD matrix from the CSV file
rmsd_df = pd.read_csv("OUTPUT_rmsd_matrix.csv", index_col=0)
rmsd_matrix = rmsd_df.values

# Convert the RMSD matrix into condensed form
condensed_rmsd = squareform(rmsd_matrix)  # Convert the square matrix to a condensed form

# === Step 2: Perform Hierarchical Clustering ===
# Perform hierarchical clustering using the condensed distance matrix
linked = sch.linkage(condensed_rmsd, method='complete')

# Display the dendrogram for selecting the cutoff
plt.figure(figsize=(10, 7))
sch.dendrogram(linked, labels=rmsd_df.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Molecule Files')
plt.ylabel('RMSD Distance')
plt.tight_layout()
plt.show()

# === Step 3: Dynamically Select RMSD Cutoff ===
# Ask the user for the RMSD cut-off
rmsd_cutoff = float(input("Enter the RMSD cut-off based on the dendrogram (look at the plot and select a reasonable value): "))

# Assign clusters based on the RMSD cut-off
labels = fcluster(linked, t=rmsd_cutoff, criterion='distance')
num_clusters = len(set(labels))
print(f"\nMolecules have been clustered into {num_clusters} groups using RMSD cut-off = {rmsd_cutoff:.2f}")

# === Step 4: Save Clusters to Directories ===
output_dir = "./clusters"
os.makedirs(output_dir, exist_ok=True)

input_dir = "./conformers_output"  # Directory containing the molecule files
cluster_dict = {}

# Copy molecule files to their respective cluster directories
for i, file in enumerate(rmsd_df.index):
    cluster_id = labels[i]
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    source_path = os.path.join(input_dir, file)
    target_path = os.path.join(cluster_dir, file)

    # Add warning for missing files
    if not os.path.exists(source_path):
        print(f"Warning: File {file} does not exist in {input_dir}. Skipping.")
        continue

    shutil.copy(source_path, target_path)

    # Track the cluster and the file
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(file)

# === Step 5: Output Cluster Summary ===
for cluster_id, files in cluster_dict.items():
    print(f"Cluster {cluster_id}: {len(files)} files")

print(f"\nCluster files have been saved in the '{output_dir}' directory.")


