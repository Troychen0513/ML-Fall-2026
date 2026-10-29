import numpy as np

expression_data = np.load("C:\\Users\\26491\\Desktop\\ML-Fall-2026\\HW4 KNN\\data2\\MERFISH\\X_expression.npz")
count_data = np.load("C:\\Users\\26491\\Desktop\\ML-Fall-2026\\HW4 KNN\\data2\\10X_cells_v3_AIBS\\X_counts.npz")
exon_data = np.load("C:\\Users\\26491\\Desktop\\ML-Fall-2026\\HW4 KNN\\data2\\SMARTer_cells_MOp\\X_exon.npz")

print(expression_data.files)  # Print the keys in the .npz file
print(count_data.files)  # Print the keys in the .npz file
print(exon_data.files)  # Print the keys in the .npz file