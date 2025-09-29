import numpy as np

split = '8020'
method = 'hclt'

filename = f"../../results/1KG/{split}/{method}/10K_{method}_{split}_samples"
filename = "/scratch2/prateek/genetic_pc_github/results/b38/8020/data/8020_test"
arr = np.loadtxt(f'{filename}.txt', dtype=int)
print(arr.shape)

num_samples, num_snps = arr.shape

train_column = np.array(["Test"] * num_samples).reshape(-1, 1)
sample_column = np.array([f"Sample{i+1}" for i in range(num_samples)]).reshape(-1, 1)
final_array = np.hstack((train_column, sample_column, arr.astype(str)))
np.savetxt(f"{filename}.hapt", final_array, fmt="%s", delimiter=" ")