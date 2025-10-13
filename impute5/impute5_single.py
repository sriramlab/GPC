# %%
import numpy as np
import pandas as pd
import subprocess
import os
import math
import sys
from sklearn.metrics import r2_score
import shutil
import tempfile

k = int(sys.argv[1]) - 1
threads = int(sys.argv[2])
train_prefix = str(sys.argv[3])
test_prefix = str(sys.argv[4])
chrnumber = int(sys.argv[5])
method_full = str(sys.argv[6])
# %%
def process_plink_data(folder, plink_file_prefix):
    vcf_file = f"{folder}/{plink_file_prefix}.vcf"

    # print("Exporting PLINK data to VCF...")
    export_command = [
        "../plink2", "--bfile", f"{folder}/{plink_file_prefix}",
        "--recode", "vcf", "--out", f"{folder}/{plink_file_prefix}",
        "--silent"
    ]
    
    if not os.path.exists(vcf_file):
        subprocess.run(export_command, check=True)

    bcftools_command = [
        "/u/scratch/p/panand2/bcftools/bcftools", "view", "-Ou", "-o", f"{folder}/{plink_file_prefix}.bcf", vcf_file
    ]

    if not os.path.exists(f"{folder}/{plink_file_prefix}.bcf"):
        subprocess.run(bcftools_command, check=True)

        # print("BCF conversion complete.")

        # print("Adding AC field..")
        subprocess.run([
        '/u/scratch/p/panand2/bcftools/bcftools', '+fill-tags', f'{folder}/{plink_file_prefix}.bcf', '-Ou', '-o', f'{folder}/{plink_file_prefix}_AC.bcf', '--', '-t', 'AN,AC'
        ])
        # print("Adding index file..")
        subprocess.run(['/u/scratch/p/panand2/bcftools/bcftools', 'index', f'{folder}/{plink_file_prefix}_AC.bcf'])

def process_plink_data_with_drop(folder, plink_file_prefix, rs_ids, temp_dir):
    vcf_file = f"{folder}/{plink_file_prefix}.vcf"

    modified_vcf_file = f"{temp_dir}/modified_test.vcf"
    with open(vcf_file, 'r') as f:
        lines = f.readlines()

    with open(modified_vcf_file, 'w') as out_file:
        for line in lines:
            if line.startswith('#'):
                out_file.write(line)
                continue
            
            fields = line.strip().split('\t')
            snp_id = fields[2]

            if snp_id in rs_ids:
                # fields[9:] = ['./.' for _ in fields[9:]]
                continue
                
            out_file.write('\t'.join(fields) + '\n')

    # print("VCF modification complete.")

    bcftools_command = [
        "/u/scratch/p/panand2/bcftools/bcftools", "view", "-Ou", '--threads', f'{threads}', "-o", f"{temp_dir}/modified_test.bcf", modified_vcf_file
    ]
    subprocess.run(bcftools_command)

    # print("BCF conversion complete.")

    # print("Adding AC field..")
    subprocess.run([
     '/u/scratch/p/panand2/bcftools/bcftools', '+fill-tags', f'{temp_dir}/modified_test.bcf', '-Ou', '-o', f'{temp_dir}/modified_test_AC.bcf', '--', '-t', 'AN,AC'
    ], stdout=subprocess.DEVNULL)
    # print("Adding index file..")
    subprocess.run(['/u/scratch/p/panand2/bcftools/bcftools', 'index', f'{temp_dir}/modified_test_AC.bcf'])

def extract_imputed_genotype_array(vcf_file, snp_set, correct_genotype_array, num_samples):
    results = {}
    
    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id in snp_set:
                c = correct_genotype_array[snp_id]

                genotype_data = fields[9:]
                genotype_array = []
                prob1 = []
                expected_counts = []
                probs = np.zeros(num_samples)
                log_probs = np.zeros(num_samples)
                log_probs_filtered = np.zeros(num_samples)

                for i, genotype in enumerate(genotype_data):
                    gt_info, _, gp_info = genotype.split(':')
                    
                    # Convert genotype format to 0, 1, or 2
                    if gt_info == "0":
                        genotype_array.append(0)
                    # elif gt_info == "0|1" or gt_info == "1|0":
                    #     genotype_array.append(1)
                    elif gt_info == "1":
                        genotype_array.append(1)
                    else:
                        genotype_array.append(-1)

                    prob_values = list(map(float, gp_info.split(',')))
                    prob_0 = prob_values[0]
                    prob_1 = prob_values[1]
                    # prob_2 = prob_values[2]
                    expected_count = prob_0 * 0 + prob_1 * 1 # + prob_2 * 2
                    expected_counts.append(expected_count)

                    prob1.append(prob_1)

                    correct = c[i]
                    probs[i] = prob_values[correct]
                    if prob_values[correct] != 0:
                        log_probs[i] = math.log(prob_values[correct])
                        log_probs_filtered[i] = math.log(prob_values[correct])
                    else:
                        prob_values = np.array(prob_values) + 0.0001
                        prob_values /= np.sum(prob_values)
                        log_probs[i] = math.log(prob_values[correct])
                        log_probs_filtered[i] = 1

                results[snp_id] = (genotype_array, prob1, expected_counts, probs, log_probs, log_probs_filtered)

    return results

def extract_test_genotype_array(vcf_file, snp_set):
    results = {}

    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id in snp_set:
                genotype_data = fields[9:]
                genotype_array = []

                for genotype in genotype_data:
                    if genotype == "0":
                        genotype_array.append(0)
                    # elif genotype == "0/1" or genotype == "1/0":
                    #     genotype_array.append(1)
                    elif genotype == "1":
                        genotype_array.append(1)
                    else:
                        genotype_array.append(-1)

                results[snp_id] = genotype_array

    return results 

def compute_r_squared_old(array1, array2):
    # Ensure both arrays are NumPy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Calculate the correlation coefficient
    correlation_matrix = np.corrcoef(array1, array2)
    correlation_xy = correlation_matrix[0, 1]
    
    # Calculate R^2
    r_squared = correlation_xy ** 2
    
    return r_squared

def find_min_max_positions_bim(bim_file):
    with open(bim_file, 'r') as f:
        positions = [int(line.split()[3]) for line in f]
    
    min_pos = min(positions)
    max_pos = max(positions)
    
    return min_pos, max_pos

def analyze_vcf(vcf_file):
    min_pos = float('inf')
    max_pos = float('-inf')
    num_snps = 0
    num_samples = 0

    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith("##"):
                continue
            
            if line.startswith("#CHROM"):
                columns = line.strip().split("\t")
                num_samples = len(columns) - 9
                continue
            
            columns = line.strip().split("\t")
            pos = int(columns[1])
            
            min_pos = min(min_pos, pos)
            max_pos = max(max_pos, pos)
            
            num_snps += 1

    return min_pos, max_pos, num_snps, num_samples
########################################################################################################################
#  %%
def main():
    temp_dir = tempfile.mkdtemp(prefix=f"temp{k}")
    # temp_dir = '/u/scratch/p/panand2/genetic_pc/test_dir'
    text_snps = "10K"
    num_latents = 128

    folder = "/u/scratch/p/panand2/genetic_pc"
    plink_file_train_prefix = train_prefix
    # plink_file_train_prefix = f"artificial_genomes/1000G_real_genomes/{text_snps}_train"
    plink_file_test_prefix = test_prefix
    chrnum = chrnumber

    vcf_file = f"{folder}/{plink_file_test_prefix}.vcf"

    idx1, idx2, num_snps, num_samples = analyze_vcf(vcf_file)
    
    print(f"Min position: {idx1}")
    print(f"Max position: {idx2}")
    print(f"Num SNPs: {num_snps}")
    print(f"Num samples: {num_samples}")

    buffer_region = f"{chrnum}:{idx1}-{idx2}"

    os.environ['BCFTOOLS_PLUGINS'] = '/scrach2/prateek/bcftools/plugins'

    process_plink_data(folder, plink_file_train_prefix)

    # bim_file = f"{folder}/{plink_file_test_prefix}.bim"
    # rs_ids = []

    # with open(bim_file, 'r') as f:
    #     for line in f:
    #         fields = line.split()
    #         rs_id = fields[1]
    #         rs_ids.append(rs_id)

    def extract_rs_ids_from_vcf(vcf_file):
        """
        Extract rsIDs from a VCF file.

        Parameters:
        - vcf_file (str): Path to the VCF file.

        Returns:
        - list: List of rsIDs extracted from the VCF file.
        """
        rs_ids = []

        with open(vcf_file, 'r') as f:
            for line in f:
                # Skip header lines starting with '##'
                if line.startswith('##'):
                    continue

                # Skip the column header line
                if line.startswith('#CHROM'):
                    continue

                # Split the VCF line into fields
                fields = line.split()

                # Extract rsID from the third column (ID column)
                rs_id = fields[2]  # This is where rsID is typically found in a VCF file
                rs_ids.append(rs_id)
        
        return rs_ids

    rs_ids = extract_rs_ids_from_vcf(f'{folder}/{plink_file_test_prefix}.vcf')

    # num_snps = len(rs_ids)
    # fam_file = f"{folder}/{plink_file_test_prefix}.fam"
    # num_samples = sum(1 for _ in open(fam_file))

    r2s = np.zeros(num_snps)
    r2s_geno = np.zeros(num_snps)
    # pseudolikelihoods = np.zeros(num_samples)
    # pseudolikelihoods_filtered = np.zeros((num_samples, num_snps))

    # for idx, rs_id in enumerate(rs_ids):
    #     print(f"Dropping SNP: {rs_id} (#{idx+1})")
    #     num = int(rs_id.split(':')[1])

    batch_size = 1  # Define batch size for SNPs to drop in each iteration

    # joints = np.zeros((num_samples, (num_snps + batch_size - 1) // batch_size))

    # for batch_idx in range(0, len(rs_ids), batch_size):

    batch_idx = k

    snp_set = rs_ids[batch_idx:batch_idx + batch_size]
    print(f"Dropping SNP set: {snp_set}")

    i = int(rs_ids[batch_idx].split(':')[1])
    j = int(rs_ids[min(batch_idx + batch_size, len(rs_ids)) - 1].split(':')[1])

    if i == j:
        j += 1
        if j == (idx2+1):
            i -= 1
            j -= 1

    process_plink_data_with_drop(folder, plink_file_test_prefix, snp_set, temp_dir)

    result = subprocess.run([
        '/u/scratch/p/panand2/impute5_v1.2.0/impute5_v1.2.0_static',
        '--h', f'{folder}/{plink_file_train_prefix}_AC.bcf',
        '--m', f'/u/scratch/p/panand2/b37_recombination_maps/chr{chrnum}.b37.gmap.gz',
        '--g', f'{temp_dir}/modified_test_AC.bcf',
        '--r', f"{chrnum}:{i}-{j}",
        '--buffer-region', buffer_region,
        '--o', f"{temp_dir}/imputed_{batch_idx}.vcf",
        '--l', f"{temp_dir}/imputed_{batch_idx}.log",
        '--haploid',
        '--threads', f'{threads}'
    ])

    test_arrays = extract_test_genotype_array(f'{folder}/{plink_file_test_prefix}.vcf', snp_set)
    results = extract_imputed_genotype_array(f'{temp_dir}/imputed_{batch_idx}.vcf', snp_set, test_arrays, num_samples)


    # r2_output_file = f"/u/scratch/p/panand2/genetic_pc/results/r2s/r2s_mask{batch_size}_{text_snps}_gan3_impute5"
    # r2_geno_output_file = f"/u/scratch/p/panand2/genetic_pc/results/r2s/r2s_geno_mask{batch_size}_{text_snps}_gan3_impute5"

    for a, snp in enumerate(snp_set):
        test_genotype_array = test_arrays[snp]
        imputed_genotype_array = results[snp][0]
        prob1 = results[snp][1]

        # print(test_genotype_array)
        # print(imputed_genotype_array)

        # print(prob1)
        # expected_counts = results[snp][2]
        # probs = results[snp][3]
        # log_probs = results[snp][4]
        # log_probs_filtered = results[snp][5]

        # print(test_genotype_array)
        # print(prob1)
        # r2_prev = compute_r_squared(test_genotype_array, imputed_genotype_array)
        # r2 = compute_r_squared(test_genotype_array, prob1)
        r2 = compute_r_squared_old(test_genotype_array, prob1)
        r2_geno = compute_r_squared_old(test_genotype_array, imputed_genotype_array)
        # print(r2_prev)
        print(r2)
        print(r2_geno)

        # Construct the line with the SNP set and scores
        # r2_line = f"{snp_set}\t{r2}\n"
        # r2_geno_line = f"{snp_set}\t{r2_geno}\n"

        # # Append to the respective output files
        # with open(r2_output_file, "a") as r2_file:
        #     r2_file.write(r2_line)

        # with open(r2_geno_output_file, "a") as r2_geno_file:
        #     r2_geno_file.write(r2_geno_line)

        # snp_index = batch_idx + a
        # r2s[snp_index] = r2
        # r2s_geno[snp_index] = r2_geno
        # pseudolikelihoods += log_probs
        # pseudolikelihoods_filtered[:, snp_index] = log_probs_filtered

        # joints[:, batch_idx // batch_size] += log_probs

        outdir = f"results/bootstrap/{method_full}_dosages/"
        os.makedirs(outdir, exist_ok=True)
        np.savetxt(f"{outdir}{snp}.txt", prob1, fmt="%.5f")

    try:
        shutil.rmtree(temp_dir)
        print(f"Directory {temp_dir} and its contents removed successfully.")
    except OSError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# %%
# print(r2s)
# print(pseudolikelihoods)

# df = pd.DataFrame(joints)
# df.to_csv(f'results/joints/joints_mask{batch_size}_{text_snps}_{num_latents}_hclt2_impute5.csv', index=False, header=False)

# df = pd.DataFrame(pseudolikelihoods_filtered)
# df.to_csv(f'results/pseudolikelihoods_mask{batch_size}_filtered_impute5.csv', index=False, header=False)
# np.savetxt(f'results/pseudolikelihoods/pseudolikelihoods_mask{batch_size}_{text_snps}_{num_latents}_hclt2_impute5', pseudolikelihoods)
# np.savetxt(f'results/r2s/r2s_mask{batch_size}_{text_snps}_{num_latents}_hclt2_impute5', r2s)
# np.savetxt(f'results/r2s/r2s_geno_mask{batch_size}_{text_snps}_{num_latents}_hclt2_impute5', r2s_geno)
# %%
