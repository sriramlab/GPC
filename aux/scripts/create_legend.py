vcf_file = "/scratch2/prateek/genetic_pc_github/results/b38/1kGP_high_coverage_Illumina.chr15.filtered_biallelic_snps_region.SNV_INDEL_SV_phased_panel_unrelated_ac20_4988.vcf"
legend_file = "/scratch2/prateek/genetic_pc_github/aux/b38_SNP.legend"

with open(vcf_file) as fin, open(legend_file, "w") as fout:
    # write header
    fout.write("id position a0 a1\n")

    for line in fin:
        if line.startswith("#"):  # skip headers
            continue
        parts = line.strip().split("\t")
        chrom = parts[0]
        pos = parts[1]
        ref = parts[3]
        alt = parts[4]  # can be comma-separated if multiple ALTs

        # handle multiple ALT alleles if needed
        for a in alt.split(","):
            snp_id = f"{chrom}:{pos}_{ref}_{a}"
            fout.write(f"{snp_id} {pos} {ref} {a}\n")