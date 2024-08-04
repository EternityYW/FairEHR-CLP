# FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records
This repository contains code for our MLHC 2024 paper [FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records](https://arxiv.org/abs/2402.00955)

## Objective
This project proposes a general framework for Fairness-aware Clinical Predictions with Contrastive Learning in EHRs. 
FairEHR-CLP operates through a two-stage process:
- Data Generation: Synthetic counterparts are created for each patient to introduce diverse demographic identities while maintaining essential health information.
- Fairness-Aware Predictions: Contrastive learning is employed to align patient representations across sensitive attributes, which are jointly optimized with an MLP classifier using a softmax layer for clinical classification tasks.

<div align="center">
    <img width="75%" alt="image" src="https://github.com/EternityYW/FairEHR-CLP/blob/main/image_sources/model_framework.png">
</div>

## Models
FairEHR-CLP: Our proposed method is implemented in patient_MIMICIII/MIMICIV/Stanford_exp.py. Note that synthetic notes are generated using Llama2_for_notes.py.

- Baselines:

  - Demographic-free Classification (DfC): This approach assumes that excluding demographic features, which are often central to socially sensitive biases, should lead to minimal differences in model performance. (See patient_MIMICIII/MIMICIV/Stanford_no_demo_exp.py)

  - Adversarial Debiasing (AdvDebias): A debiasing strategy tailored for EHR that simultaneously trains a classifier and an adversary model to neutralize bias. (Model link: https://github.com/yangjenny/adversarial_learning_bias_mitigation)

  - Fair Patient Model (FPM): Employs a Stacked Denoising Autoencoder and a weighted reconstruction loss to ensure equitable patient representations. (https://www.sciencedirect.com/science/article/pii/S1532046423002654)

  - RoBERTa-large: A widely-used embedding method for general applications. (Model link: https://huggingface.co/FacebookAI/roberta-large)

  - ClinicalBERT: A healthcare-specific embedding method designed for medical applications. (Model link: https://huggingface.co/medicalai/ClinicalBERT)
 
## Datasets
We evaluate our proposed framework using three EHR datasets: STAnford medicine Research data Repository (STARR) from Stanford Medicine, MIMIC-III, and MIMIC-IV. The focus is on surgical patients aged 50 or older, a cohort prone to age-related biases like impaired cognition. To avoid patient data overlap, we use the MIMIC-III Clinical Database CareVue subset. The study targets three tasks: classifying delirium, OUD, and 30-day readmission, chosen for their impact on postoperative care and patient safety. Demographic indicators are excluded from clinical notes to concentrate on health conditions. We extract patient data from a 24-hour postoperative period and use MICE imputation to address missing values. Each task is approached as a binary classification problem. The table below represents the class distribution for each task.

<div align="center">
    <img width="75%" alt="image" src="https://github.com/EternityYW/FairEHR-CLP/blob/main/image_sources/data_overview.png">
</div>

Please get in touch with the corresponding author for sample data inputs.

## Getting started
We use STARR dataset as an demonstration example (patient_Stanford_exp.py).

Step 1: Load necessary packages

```
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pdb
import nlpaug.augmenter.word as naw
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from scipy.stats import wasserstein_distance
```

Step 2: Load datasets and generate synthetic counterparts

```
df = pd.read_csv("structured_Stanford.csv")
df_notes = pd.read_csv("unstructured_Stanford.csv")

df_demographics = df[['pat_deid', 'sex', 'ethnic_group', 'race', 'age', 'surg_family', 'product_type', 'tobacco_user', 'readmission_30_days_label']]
df_demographics = df_demographics.drop_duplicates()
df_longitudinal = df[['pat_deid', 'Heart Rate', 'Pulse', 'Resp', 'SpO2','Temp', 'Systolic_BP', 'Diastolic_BP', 'ALT (SGPT), Ser/Plas', 'Albumin, Ser/Plas', 'Anion Gap', 'BUN, Ser/Plas', 'CO2, Ser/Plas', 'Calcium, Ser/Plas', 'Chloride, Ser/Plas', 'Creatinine, Ser/Plas', 'Glucose, Ser/Plas', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Platelet count', 'Potassium, Ser/Plas', 'RBC', 'RDW', 'Sodium, Ser/Plas', 'WBC']]
...
train_dataset = Subset(PatientPairDataset(real_data, synthetic_data, labels, use_synthetic=True), train_indices)
test_dataset = Subset(PatientPairDataset(real_data, synthetic_data, labels, use_synthetic=False), test_indices)
)
```

Step 3: Model training and evaluation

```
model = FairnessAwareModel().to(device)
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    ...
    model.eval()
    ...
    np.save(f'{epoch_str}_task_test_ground_truth.npy', np.array(test_ground_truth))
    np.save(f'{epoch_str}_task_test_predictions.npy', np.array(test_predictions))
```
Step 4: Obtain the output .npy for ground truth and model predictions for further performance and fairness evaluation.

Step 5: Feature Extraction and Fairness Analysis

Please see feature_extraction_and_fairness_analysis.ipynb for how features from the STARR dataset are extracted and for fairness evaluation using EO and EDDI scores.


## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@article{wang2024fairehr,
  title={FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records},
  author={Wang, Yuqing and Pillai, Malvika and Zhao, Yun and Curtin, Catherine and Hernandez-Boussard, Tina},
  journal={arXiv preprint arXiv:2402.00955},
  year={2024}
}
```
