import numpy as np
import os

def vcf_to_haplotype_array(vcf_file):
    haplotypes = []

    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                header = line.strip().split('\t')
                continue

            fields = line.strip().split('\t')
            genotypes = fields[9:]  # genotype columns

            genotype_row = []
            for gt in genotypes:
                # Extract phased genotype: e.g. "0|0", "1|1"
                # Some VCFs include additional info after genotype like "0|0:35:99", so split by ":" first
                gt_simple = gt.split(':')[0]

                # Map "0|0" -> 0, "1|1" -> 1, else raise or assign missing
                if gt_simple == '0':
                    genotype_row.append(0)
                elif gt_simple == '1':
                    genotype_row.append(1)
                else:
                    # Handle missing or other genotypes
                    # For now, treat all others as missing (e.g., np.nan or -1)
                    genotype_row.append(-1)  # or np.nan if using float array later

            haplotypes.append(genotype_row)

    haplotype_array = np.array(haplotypes)

    # Transpose to get samples as rows, SNPs as columns
    return haplotype_array.T

def create_vcf_from_legend(haplotype_array, snp_legend_file, output_file='output.vcf', reference='hg37'):
    """
    Create a VCF file using a haplotype array and a SNP legend file.

    Parameters:
    - haplotype_array (2D numpy array): Array where rows represent SNPs and columns represent samples.
    - snp_legend_file (str): Path to the SNP legend file with columns: id, position, a0, a1.
    - output_file (str): Path to the output VCF file.
    - reference (str): Reference genome identifier (e.g., 'hg37').
    """
    # Read the SNP legend file
    snp_data = []
    with open(snp_legend_file, 'r') as legend_file:
        next(legend_file)  # Skip header line
        for line in legend_file:
            parts = line.strip().split()
            snp_id = parts[0].split('_')[0]
            position = int(parts[1])
            ref_allele = parts[2]
            alt_allele = parts[3]
            chrom = snp_id.split(':')[0]

            # Remove "chr" prefix if present
            if chrom.lower().startswith("chr"):
                chrom = chrom[3:]
                snp_id = snp_id.replace("chr", "", 1)  # remove only leading "chr"

            snp_data.append((chrom, position, snp_id, ref_allele, alt_allele))
    
    # Validate haplotype array size matches the SNP legend
    if len(snp_data) != haplotype_array.shape[0]:
        raise ValueError("Number of SNPs in the haplotype array does not match the SNP legend file.")

    # Create sample names
    num_samples = haplotype_array.shape[1]
    samples = [f"Sample_r{i+1}" for i in range(num_samples)]

    # Create VCF header
    vcf_header = [
        "##fileformat=VCFv4.2",
        "##source=my_haplotype_data",
        f"##reference={reference}",
        "##contig=<ID=15>",
        "##INFO=<ID=PR,Number=0,Type=Flag,Description=\"Variant is present in reference panel\">",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)
    ]

    # Create VCF body
    vcf_body = []
    for i, (chrom, position, snp_id, ref_allele, alt_allele) in enumerate(snp_data):
        # Format genotypes for each sample (convert float to integer for 0/1)
        genotypes_str = "\t".join([(str(int(g))) for g in haplotype_array[i]])
        # genotypes_str = "\t".join([(str(int(g)) + "|" + str(int(g))) for g in haplotype_array[i]])
        
        # VCF data line
        vcf_line = f"{chrom}\t{position}\t{snp_id}\t{ref_allele}\t{alt_allele}\t.\t.\tPR\tGT\t{genotypes_str}"
        vcf_body.append(vcf_line)

    # Write to VCF file
    with open(output_file, 'w') as f:
        for line in vcf_header:
            f.write(line + "\n")
        for line in vcf_body:
            f.write(line + "\n")

def bootstrap_and_write_vcfs(haplotype_array, snp_legend_file, outdir, n_bootstraps=10):
    os.makedirs(outdir, exist_ok=True)

    num_samples = haplotype_array.shape[0]

    for i in range(n_bootstraps):
        print(i)
        # Sample with replacement along the rows (samples)
        indices = np.random.choice(num_samples, size=num_samples, replace=True)
        bootstrapped_data = haplotype_array[indices]

        # Create a unique output VCF file for each bootstrap replicate
        output_vcf_file = os.path.join(outdir, f"bootstrap_{i+1}.vcf")
        create_vcf_from_legend(bootstrapped_data.T, snp_legend_file, output_file=output_vcf_file)

        # Save indices used for this bootstrap
        index_file = os.path.join(outdir, f"indices_{i+1}.txt")
        np.savetxt(index_file, indices, fmt='%d')

if __name__ == "__main__":
    # Load haplotype matrix from VCF
    haplotype_array = vcf_to_haplotype_array('/scratch2/prateek/genetic_pc_github/results/b38/8020/data/b38_test.vcf')

    snp_legend_file = '/scratch2/prateek/genetic_pc_github/aux/b38_SNP.legend'
    outdir = '/scratch2/prateek/genetic_pc_github/results/b38/8020/data/test_bootstraps'

    # Generate and write 10 bootstrapped VCFs
    bootstrap_and_write_vcfs(haplotype_array, snp_legend_file, outdir, n_bootstraps=10)