# FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records
This repository contains code for our MLHC 2024 paper [FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records](https://arxiv.org/abs/2402.00955)

## Objective
This project proposes a general framework for Fairness-aware Clinical Predictions with Contrastive Learning in EHRs. 
FairEHR-CLP operates through a two-stage process:
- Data Generation: Synthetic counterparts are created for each patient to introduce diverse demographic identities while maintaining essential health information.
- Fairness-Aware Predictions: Contrastive learning is employed to align patient representations across sensitive attributes, which are jointly optimized with an MLP classifier using a softmax layer for clinical classification tasks.

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
