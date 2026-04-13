# Explainable Novel Category Discovery in Semantic Concept Space

## Abstract
Novel category discovery requires learning decision rules that induce meaningful partitions over unlabeled data while transferring information from labeled classes. Existing approaches rely on highly expressive latent feature spaces, allowing many admissible partitions that separate data without semantic structure. We propose an interpretable framework for novel category discovery that performs representation learning and category discovery directly in a structured semantic concept space. The model learns concept detectors whose activations define an explicit intermediate representation shared by labeled and unlabeled samples. Concept supervision is obtained without manual annotation via alignment with vision–language similarity priors from pretrained multimodal models. We formulate discovery as a unified learning objective over concept representations and provide a theoretical analysis showing that concept bottlenecks impose a strict restriction on the hypothesis space of novel category discovery models. This restriction eliminates a large class of semantically entangled classifiers and constrains the set of induced partitions to those expressible through interpretable concept coordinates. Empirical results on standard benchmarks demonstrate that this hypothesis space restriction preserves competitive discovery performance while yielding intrinsically interpretable novel categories.

---

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Training Pipeline](#training-pipeline)
  - [Stage 1: Pretraining](#stage-1-pretraining)
  - [Stage 2: Concept Layer Training](#stage-2-concept-layer-training)
  - [Stage 3: Novel Class Discovery](#stage-3-novel-class-discovery)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning 2.0+
- CUDA 11.x
- CLIP (OpenAI)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/repo.git
cd /path-to-project-directory

# Run the environment setup script
bash env-setup.sh

# Activate the environment
conda activate CB-NCD
```

## Datasets

Our code supports the following datasets:

| Dataset | Total Classes | Labeled | Unlabeled | Architecture | Image Size |
|---------|---------------|---------|-----------|--------------|------------|
| CIFAR-10 | 10 | 5 | 5 | ResNet-18 | 32x32 |
| CIFAR-100 | 100 | 80 | 20 | ResNet-18 | 32x32 |
| ImageNet | 912 | 882 | 30 | ResNet-18 | 224x224 |
| CUB-200-2011 | 200 | 170 | 30 | ResNet-50 | 224x224 |


### Dataset Preparation

**CIFAR-10/100**: Downloaded automatically with `--download` flag.

**Imagenet**: Needs to download Imagenet1k

**CUB-200-2011**:
```bash
# Download and extract
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz -C datasets/

# After that run this to prepare original CUB concept vocabulary or combined with generated one.
python create_cub_concepts.py
```

---

## Training Pipeline

xNCD follows a three-stage training pipeline. For complete configurations including all hyperparameters, refer to the job files:

- CIFAR-10: `job.yml`
- CIFAR-100: `job_cifar100.yml`
- CUB-200: `job_cub.yml`
- Imagenet: 'job_imagenet.yml

**Runing the .yml file will run the full pipleline for its respctive dataset. Adjust the path and hyperparameters as per your configuration**

## Evaluation Metrics

CB-NCD reports the following metrics:

| Metric | Description | Computed On |
|--------|-------------|-------------|
| **Labeled Accuracy** | Classification accuracy | Labeled (known) classes |
| **Unlabeled Accuracy** | Clustering accuracy via Hungarian matching | Unlabeled (novel) classes |
| **All Accuracy** | Overall accuracy (incremental setting) | All classes |
| **NMI** | Normalized Mutual Information | Novel classes |
| **ARI** | Adjusted Rand Index | Novel classes |

Metrics are logged to Weights & Biases:
- `lab/test/acc` - Labeled class accuracy
- `unlab/test/acc/best` - Best head clustering accuracy on novel classes
- `unlab/test/nmi/best` - Best head NMI
- `unlab/test/ari/best` - Best head ARI

---

## Visualization

After training, use the provided Jupyter notebooks for visualization:

- `cifar100_ncd_visualization.ipynb` - CIFAR-100 cluster and concept analysis
- `cub_ncd_visualization.ipynb` - CUB-200 fine-grained concept visualization


---

## Running on Kubernetes

Job configuration files are provided for running on Kubernetes clusters:

```bash
# Submit CIFAR-10 job
kubectl apply -f job.yml

# Submit CIFAR-100 job
kubectl apply -f job_cifar100.yml

# Submit CUB-200 job
kubectl apply -f job_cub.yml

# Check job status
kubectl get jobs -n your-namespace
kubectl logs job/cb-ncd-cifar100 -n your-namespace
```

--
## Cluster-Level Signatures
<p align="center">
  <img src="output/cluster_N2_profile_balanced.png" width="49%"/>
  <img src="output/cluster_N4_profile_balanced.png" width="49%"/>
</p>

-- 
## Instance-Level Explanations

<p align="center">
  <img src="output/sample_2_image.png" width="12%"/>
  <img src="output/sample_2_concepts.png" width="35%"/>
  &nbsp;&nbsp;&nbsp;
  <img src="output/sample_7_image.png" width="12%"/>
  <img src="output/sample_7_concepts.png" width="35%"/>
</p># proto-ncd
