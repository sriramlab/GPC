# GPC: Deep generative model of genetic variation data improves imputation accuracy in private populations

Official repository for artificial genome generation and imputation using GPC.

This repository is actively being updated.

All code in the `plots/structure/` directory is adopted and modified from this paper: [Deep convolutional and conditional neural networks for large-scale genomic data generation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011584#sec002).

---

## Installation

### Install PyJuice genetic pc branch from source

```bash
git clone --branch genetic-pc --single-branch https://github.com/Tractables/pyjuice.git
cd pyjuice
pip install -e .
```
### Clone this repository
```bash
git clone https://github.com/prateekanand2/genetic_pc.git
```

### Train a small model on the 805 SNP data and generate samples
```bash
cd genetic_pc/pc/demo
python3 train_demo.py
python3 generate_demo.py
```

### Visualize PCA
```bash
pc/demo/plot.ipynb
```
