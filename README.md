# Hyperbolic and Euclidean Mixture Knowledge Graph Embedding for Rare Disease Diagnosis (Fork)

This repository is a fork of the original [Hyperbolic Knowledge Graph Embedding](https://github.com/HazyResearch/KGEmb) implementation, extending it to simulate in-silico differential diagnosis of rare Mendelian diseases.

## 1. Research Objectives and Motivations

The diagnosis of rare diseases presents a profound challenge in modern medicine. Despite impacting a meager 5.9% of the global population [1], the collective burden is substantial: the diagnostic "odyssey" can take up to five years, and typically involves consultations with at least five different doctors and multiple misdiagnoses [2]. This delay is rooted in the clinical scarcity of cases, which complicates the generalization of symptoms. Diagnosis therefore requires exploration of diverse biomedical knowledge across medical domains. However, biomedical data remains highly fragmented and domain-specific.

A comprehensive mapping of relationships among diseases, drugs, phenotypes, and molecular entities is critical to enabling systematic, context-aware predictions, ranging from phenotyping [3, 4] to drug repurposing [5, 6]. To organize this immense knowledge, graphs have emerged as the most suitable data structure. Graph topology acts as a structural constraint that explicitly defines how elements interact, effectively reducing the system's degrees of freedom. Beyond a theoretical framework, graphs provide the mathematical foundation to model complex system behaviors, so that learning these structures allows for simulating reasoning aware of multi-step underlying interactions (e.g., protein pathways, drug mechanisms).

However, Machine Learning on these discrete structures requires mapping nodes (i.e., discrete non-Euclidean entities) to continuous vector representations that typically reside in Euclidean latent space. However, this choice of latent geometry often misaligns with the intrinsic structure of biomedical data, which is heavily grounded in *Ontologies* (e.g., MONDO, GO). These are Directed Acyclic Graphs (DAGs) that always express a strong taxonomical backbone, typically enriched by non-hierarchical relations that describe properties between objects. As established by Bourgain et al., forcing these non-Euclidean hierarchical structures into a flat Euclidean geometry inevitably leads to significant distortion and loss of information [7, 8].

Hyperbolic space, instead, grows exponentially, and therefore it can easily accommodate exponentially growing tree structures, allowing for low-distortion embeddings even at low dimensions [9, 10].  Yet, creating a comprehensive Biomedical Knowledge Base requires connecting distinct ontologies together. Building such multimodal Knowledge Graphs (KGs) involves combining deep ontological hierarchies with "transversal" relations (e.g., *drug treats disease*) that may exhibit logical patterns (e.g., symmetry, translation, or cyclicity) traditionally better suited to Euclidean geometry. To address this geometric dichotomy, we introduce a novel hybrid framework that successfully performs *in silico* differential diagnosis of rare Mendelian diseases on custom built patient-integrated Knowledge Graph, demonstrating that proper topological alignment drives clinical precision.


## 2. New Additions

This fork introduces several key additions to the original repository:

*   **New Dataset**: A custom Knowledge Graph that integrates patient phenotypes with molecular ontologies. The datasets can be found raw under `raw_data/edges_filtered.csv` and `raw_data/nodes_filtered.csv` 
*   **Hybrid Model (`AttEH`)**: A novel mixture model that combines Euclidean and Hyperbolic geometries to better represent the mixed-domain nature of the biomedical KG. The implementation can be found in `KGEmb/models/mixture.py`. 
*   **Disease-Only MRR**: A new evaluation metric that calculates the Mean Reciprocal Rank (MRR) considering only disease entities as potential candidates. This is implemented through a `disease_mask` found in:
    *   **Model layer** (`KGEmb/models/`): Each base class (`BaseE`, `BaseC`, `BaseH`, and `AttEH`) accepts `disease_ids` and a `restrict_to_diseases` flag, and registers a boolean `disease_mask` buffer. During evaluation, `get_ranking()` in `KGEmb/models/base.py` uses this mask to set non-disease scores to `-1e6`, effectively restricting the ranking to disease entities only.
    *   **Metrics layer** (`KGEmb/metrics/ancestorsMRR.py`): The same mask is applied when computing the ancestor-weighted MRR, ensuring consistency between standard and hierarchy-aware evaluation.
    *   **Orchestration** (`KGEmb/test.py`): Disease entities are identified from `nodes_filtered.csv` (type `"Disease"`) and mapped to model indices via `ent2idx.pickle`. Two model instances are created from the same checkpoint: one with `restrict_to_diseases=True` (disease-only metrics) and one with `restrict_to_diseases=False` (full-KG metrics).
*   **Ancestor MRR**: A metric that weights the MRR score based on the ontological similarity of diseases, rewarding predictions that are "closer" in the ontology to the true disease. The implementation is in `KGEmb/metrics/ancestorsMRR.py`.

## 3. How to Run the Code

### 3.1. Environment Setup

**Using Conda:**

```bash
conda create -n hyp_kg_env python=3.7 #or higher
conda activate hyp_kg_env
pip install -r KGEmb/requirements.txt
```

**Using venv:**

```bash
virtualenv -p python3.7 hyp_kg_env
source hyp_kg_env/bin/activate
pip install -r KGEmb/requirements.txt
```


NOTE: you must source the `set_env.sh` script from the **root of the repository** (outside the `KGEmb` directory) to set the necessary environment variables:

```bash
source KGEmb/set_env.sh
```

### 3.2. Training a Model

To train a model, use the `run.py` script. For example, to train the new `AttEH` model:

```bash
python KGEmb/run.py \
    --dataset dataset_name \
    --model AttEH \
    --rank 500 \
    --max_epochs 100 \
    --batch_size 1000 \
    --learning_rate 0.1 \
    --neg_sample_size 50 \
    --regularizer N3 \
    --reg 0
```

### 3.3. Testing a Model

To test a trained model, use the `test.py` script. The `--all` flag runs a comprehensive evaluation:
-  including splits by disease frequency and the official MONDO rare disease subset.

```bash
python KGEmb/test.py \
    --model_dir KGEmb/logs/<your_model_dir> \
    --all
```

## 4. Arguments

### 4.1. Training Arguments (`run.py`)

```
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10,KG_ALL,KG_PD,KG_GO,KG_F,KG_F_HasD_noV,KG_F_HasD_V,KG_filtered_v8,KG_filtered_v9,KG_filtered_v10,KG_filtered_v11,KG_filtered_v12,KG_filtered_v13}]
              [--model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE,AttEH}]
              [--regularizer {N3,F2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SparseAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]
              [--no_valid]
```

### 4.2. Testing Arguments (`test.py`)

```
usage: test.py [-h] --model_dir MODEL_DIR [--split_by_frequency]
               [--split_by_official_rare] [--all]
               [--ancestors_path ANCESTORS_PATH]
               [--rare_subset_path RARE_SUBSET_PATH]
```

## 5. Advanced Usage

### 5.1. Weights & Biases (wandb)

To use `wandb` for logging, set your API key as an environment variable before running the training script:

```bash
export WANDB_API_KEY="your_wandb_api_key"
python KGEmb/run.py ...
```

The script will automatically detect the API key and log the run to your `wandb` project.

### 5.2. Running on Slurm Clusters

The `slurms/` directory contains example scripts for running training and hyperparameter optimization on Slurm-based clusters.

**Training a single model:**

The `KGEmb_runpy.slurm` script can be used to submit a training job. You need to edit the script to set your account details and environment paths.

```bash
# Example usage:
sbatch slurms/KGEmb_runpy.slurm --dataset KG_filtered_v12 --model AttEH --rank 500
```

**Hyperparameter Optimization:**

The `comparative.slurm` script can be used to run a hyperparameter search using one of the scripts in `KGEmb/hyperp_search/`.

```bash
# Example usage:
sbatch slurms/comparative.slurm KGEmb/hyperp_search/hpo_search.py 32 16
```

This command would run the `hpo_search.py` script for rank 32 with 16 trials.


---

### Bibliography

**[1]** Y. Zhao, Z. S. Y. Wong, and K. L. Tsui. A framework of rebalancing imbalanced healthcare data for rare events’ classification: A case of look-alike sound-alike mix-up incident detection. *Journal of Healthcare Engineering*, 2018:6275435, 2018. doi: 10.1155/2018/6275435.

**[2]** X. Chen, X. Mao, Q. Guo, L. Wang, S. Zhang, and T. Chen. Rarebench: Can llms serve as rare diseases specialists? *arXiv preprint, arXiv:2402.06341*, 2024. URL: [https://arxiv.org/abs/2402.06341](https://arxiv.org/abs/2402.06341).

**[3]** Y. A. Lussier and Y. Liu. Computational approaches to phenotyping: high-throughput phenomics. *Proceedings of the American Thoracic Society*, 4(1):18–25, 2007.

**[4]** Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu. Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1):6085, 2018.

**[5]** Z. Wu, Y. Wang, L. Chen, and H. Lin. Network-based drug repositioning. *Molecular BioSystems*, 9(6):1268–1281, 2013. doi: 10.1039/C3MB00002J.

**[6]** J. T. Dudley et al. Exploiting drug–disease relationships for computational drug repositioning. *Science Translational Medicine*, 3(96):96ra76, 2011.

**[7]** J. Bourgain. On lipschitz embedding of finite metric spaces in hilbert space. *Israel Journal of Mathematics*, 52(1-2):46–52, 1985.

**[8]** S. Ben-David, N. Eiron, and H. U. Simon. Limitations of learning via embeddings in euclidean half-spaces. *Journal of Machine Learning Research*, 3(Nov):441–461, 2002.

**[9]** M. Nickel and D. Kiela. Poincaré embeddings for learning hierarchical representations. In *Advances in Neural Information Processing Systems*, volume 30, 2017.

**[10]** B. P. Chamberlain, J. Clough, and M. P. Deisenroth. Neural embeddings of graphs in hyperbolic space. In *Proceedings of the 13th International Workshop on Mining and Learning with Graphs (MLG)*, pages 1–7, 2017. doi: 10.48550/arXiv.1705.10359.

**[11]** S. Chandak et al. Primekg: A precision medicine knowledge graph. *Scientific Data*, 10:19, 2023. doi: 10.1038/s41597-023-01960-3.
