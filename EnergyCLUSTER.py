import os
import subprocess
import pandas as pd

# Parameters
clusters_dir = "./clusters"  # Directory containing the cluster directories
output_excel = "cluster_statistics.xlsx"  # Output Excel file

# Function to calculate the energy of a molecule
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
                return float(energy)  # Return as float

        # If no energy value found, return None
        print(f"Energy not found in the output for {molecule_file}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error calculating energy for {molecule_file}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for {molecule_file}: {e}")
        return None

# Dictionary to store statistics for each cluster
cluster_stats = []

# Traverse through each cluster directory
for cluster in os.listdir(clusters_dir):
    cluster_path = os.path.join(clusters_dir, cluster)
    if os.path.isdir(cluster_path):
        file_count = 0
        total_energy = 0.0
        energy_count = 0
        lowest_energy = float('inf')
        lowest_energy_file = None

        # Traverse through each molecule file in the cluster directory
        for molecule_file in os.listdir(cluster_path):
            molecule_path = os.path.join(cluster_path, molecule_file)
            if molecule_file.endswith(('.mol', '.sdf', '.pdb')):  # Assuming supported formats
                file_count += 1
                energy = calculate_energy(molecule_path)
                if energy is not None:
                    total_energy += energy
                    energy_count += 1

                    # Check for the lowest energy
                    if energy < lowest_energy:
                        lowest_energy = energy
                        lowest_energy_file = molecule_file

        # Calculate the average energy if at least one valid energy was found
        avg_energy = total_energy / energy_count if energy_count > 0 else None

        # Add the cluster statistics to the list
        cluster_stats.append({
            'Cluster': cluster,
            'Number of Files': file_count,
            'Average Energy (kcal/mol)': avg_energy,
            'Lowest Energy Conformer': lowest_energy_file,
            'Lowest Energy (kcal/mol)': lowest_energy if lowest_energy_file else None
        })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(cluster_stats)

# Save the DataFrame to an Excel file
df.to_excel(output_excel, index=False)

print(f"Cluster statistics have been saved to {output_excel}")
