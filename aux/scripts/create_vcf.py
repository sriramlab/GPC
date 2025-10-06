import numpy as np
import random
import vcfpy
import argparse

def create_vcf_from_legend(haplotype_array, snp_legend_file, output_file='output.vcf', reference='hg38'):
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
    samples = [f"Sample_{i+1}" for i in range(num_samples)]

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
        # genotypes_str = "\t".join([(str(int(g))) for g in haplotype_array[i]])                          # for creating test VCF files
        genotypes_str = "\t".join([(str(int(g)) + "|" + str(int(g))) for g in haplotype_array[i]])    # for creating ref. panel VCF files that are diploid homozygous to work with Impute5 haploid mode
        
        # VCF data line
        vcf_line = f"{chrom}\t{position}\t{snp_id}\t{ref_allele}\t{alt_allele}\t.\t.\tPR\tGT\t{genotypes_str}"
        vcf_body.append(vcf_line)

    # Write to VCF file
    with open(output_file, 'w') as f:
        for line in vcf_header:
            f.write(line + "\n")
        for line in vcf_body:
            f.write(line + "\n")

method = "wgan"
split = "afr"
file = f"../../results/b38/{split}/{method}/b38_{method}_{split}_samples"
hap_file = f"{file}.txt"
vcf_file = f"{file}.vcf"

haplotype_array = np.loadtxt(hap_file)
snp_legend_file = '../b38_SNP.legend'
output_vcf_file = vcf_file

create_vcf_from_legend(haplotype_array.T, snp_legend_file, output_file=output_vcf_file)