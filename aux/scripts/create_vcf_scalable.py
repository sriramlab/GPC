import os
import tempfile
import shutil

def create_vcf_from_legend_scalable(
    hap_file,
    snp_legend_file,
    output_file,
    reference="hg38",
    block_flush=1000,
):
    # -----------------------------
    # Read legend
    # -----------------------------
    snp_data = []
    with open(snp_legend_file) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            snp_id = parts[0].split("_")[0]
            pos = int(parts[1])
            ref, alt = parts[2], parts[3]

            chrom = snp_id.split(":")[0]
            if chrom.lower().startswith("chr"):
                chrom = chrom[3:]
                snp_id = snp_id.replace("chr", "", 1)

            snp_data.append((chrom, pos, snp_id, ref, alt))

    num_snps = len(snp_data)

    # -----------------------------
    # Temp directory for SNP buffers
    # -----------------------------
    tmpdir = tempfile.mkdtemp(prefix="vcf_tmp_", dir="/scratch2/prateek")

    try:
        snp_files = [
            open(os.path.join(tmpdir, f"snp_{i}.gt"), "w")
            for i in range(num_snps)
        ]

        # -----------------------------
        # Stream hap file
        # -----------------------------
        num_samples = 0
        with open(hap_file) as hap:
            for line_idx, line in enumerate(hap):
                alleles = line.strip().split()
                if len(alleles) != num_snps:
                    raise ValueError("SNP count mismatch")

                for i, a in enumerate(alleles):
                    snp_files[i].write(f"\t{a}|{a}")
                    # snp_files[i].write(f"\t{a}")

                num_samples += 1
                if num_samples % block_flush == 0:
                    for f in snp_files:
                        f.flush()
                    print(f"Processed {num_samples} samples")

        for f in snp_files:
            f.close()

        # -----------------------------
        # Write final VCF
        # -----------------------------
        with open(output_file, "w") as out:
            samples = [f"Sample_{i+1}" for i in range(num_samples)]

            out.write("##fileformat=VCFv4.2\n")
            out.write("##source=my_haplotype_data\n")
            out.write(f"##reference={reference}\n")
            out.write("##contig=<ID=22>\n")
            out.write(
                "##INFO=<ID=PR,Number=0,Type=Flag,"
                "Description=\"Variant is present in reference panel\">\n"
            )
            out.write(
                "##FORMAT=<ID=GT,Number=1,Type=String,"
                "Description=\"Genotype\">\n"
            )
            out.write(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(samples)
                + "\n"
            )

            for i, (chrom, pos, snp_id, ref, alt) in enumerate(snp_data):
                out.write(
                    f"{chrom}\t{pos}\t{snp_id}\t{ref}\t{alt}\t.\t.\tPR\tGT"
                )
                with open(os.path.join(tmpdir, f"snp_{i}.gt")) as f:
                    shutil.copyfileobj(f, out)
                out.write("\n")

    finally:
        shutil.rmtree(tmpdir)

# -----------------------------
# Same invocation pattern as before
# -----------------------------
file = "/scratch2/prateek/DeepLearningImputation/data/UKBB/wes/train_filtered_no_mono"

hap_file = f"{file}.txt"
vcf_file = f"{file}.vcf"
snp_legend_file = "/scratch2/prateek/DeepLearningImputation/data/UKBB/wes/chr22_wes_no_mono_SNP.legend"

create_vcf_from_legend_scalable(
    hap_file,
    snp_legend_file,
    vcf_file,
    reference="hg38",
)