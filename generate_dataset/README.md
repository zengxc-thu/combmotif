## Dataset preprocessing

### Half_life dataset

We use the dataset collected by Agarwal,et al<sup><a href="#ref1">1</a></sup>. The raw data can be download [here](https://zenodo.org/records/6326409). 

We preprocessed the dataset provided by Agarwal in **generate_halflife_data.py**. Before processing, plz make sure that you have installed [basenji](https://github.com/calico/basenji).

### MRL dataset

We use the dataset collected by Sample,et al<sup><a href="#ref2">2</a></sup>. The data can be download [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002). 

Our noAUG dataset is generated from **GSM3130435_egfp_unmod_1.csv** by **generate_noAUG_dataset.py**

### Motif cluster

Many motifs in the motif database are highly similar. We will merge these motifs for better presentation of the results. The meme motif database can be download from [MEME](https://meme-suite.org/meme/meme-software/Databases/motifs/motif_databases.12.24.tgz). We consiter both RBP and microRNA motif database, namely the Ray2013_rbp_Homo_sapiens.meme and Homo_sapiens_hsa.meme. Running the following command to obtain motif clusters. Final results will be saved into data dir.

```bash
python get_adj.py
python motif_cluster.py
```



## References
1. <p name = "ref1">Agarwal, Vikram, and David R. Kelley. "The genetic and biochemical determinants of mRNA degradation rates in mammals." Genome Biology 23.1 (2022): 245.</p>
2. <p name = "ref2">Sample, Paul J., et al. "Human 5′ UTR design and variant effect prediction from a massively parallel translation assay." Nature biotechnology 37.7 (2019): 803-809.</p>