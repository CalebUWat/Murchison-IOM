import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Load dataset
totalConf = pd.read_csv("TotalEnergiesAllConfs.csv")

# Create the histogram with seaborn
g = sns.displot(totalConf, x="Energy (kcal/mol)", binwidth=12, height=6, aspect=1.5)

# Access the matplotlib Axes object for further customization
ax = g.axes[0, 0]  # Corrected access to the Axes object

# Change the color of one bar (for example, the 13th bar)
for i, patch in enumerate(ax.patches):
    if i == 4:
        patch.set_facecolor('yellow')
    elif i ==22:
        patch.set_facecolor('lightcoral')
    else:
        patch.set_facecolor('lightblue')  # Set all other bars to light blue

# Show plot
plt.show()

