import gzip

vcf_file = "/scratch2/prateek/genetic_pc_github/results/1KG/b38/1kGP_high_coverage_Illumina.chr15.filtered_biallelic_snps_region.SNV_INDEL_SV_phased_panel.vcf.gz"
output_file = "../b38_real.hapt"

with gzip.open(vcf_file, "rt") as f:
    samples = []
    haplotypes = {}

    for line in f:
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            parts = line.strip().split("\t")
            samples = parts[9:]  # sample IDs
            # initialize haplotype lists
            for s in samples:
                haplotypes[s + "_A"] = []
                haplotypes[s + "_B"] = []
            continue

        parts = line.strip().split("\t")
        genotypes = parts[9:]

        for s, gt in zip(samples, genotypes):
            gt = gt.split(":")[0]  # take GT only
            if "|" not in gt:
                raise ValueError(f"Unphased genotype found: {gt}")
            a1, a2 = gt.split("|")
            haplotypes[s + "_A"].append(a1)
            haplotypes[s + "_B"].append(a2)

# write output
with open(output_file, "w") as out:
    for s in samples:
        out.write("Real " + s + "_A " + " ".join(haplotypes[s + "_A"]) + "\n")
        out.write("Real " + s + "_B " + " ".join(haplotypes[s + "_B"]) + "\n")

print(f"Written haplotypes to {output_file}")