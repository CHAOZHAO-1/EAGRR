# 📌 EAGRR  

![GitHub visitors](https://visitor-badge.laobi.icu/badge?page_id=CHAOZHAO-1.EAGRR&color=blue&style=flat-square)

---

## 📰 Paper Information

**Title**  
**Environment-Aware Graph Relational Reasoning for Interpretable and Generalizable Mechanical Transmission System Distributed Fault Diagnosis**

**Journal**  
*Expert Systems with Applications* (2025)

**Paper Link**  
🔗 https://www.sciencedirect.com/science/article/pii/S0957417425045774

---

## ✨ Abstract

In recent years, numerous fault diagnosis models have been developed to monitor the health status of mechanical transmission systems under dynamic environments. However, most existing approaches focus on point-to-point monitoring of individual components, adopting a local perspective while overlooking the coupling relationships among multiple components. In practice, faults originating in a single component may propagate to adjacent components through vibration transmission paths, which can lead to misdiagnosis or misleading interpretations. To address this challenge, this paper proposes an environment-aware graph relational reasoning (EAGRR) framework based on a discover–evaluate–refine paradigm for comprehensive system-level health monitoring. The proposed framework first identifies diagnostically significant sensors and captures collaborative signal variations to construct stable relational subgraphs. Then, samples from different source domains are mutually perturbed to simulate working condition variations and evaluate the robustness of the learned subgraph structures. This evaluation feedback further guides the refinement of subgraph discovery, ensuring the extraction of environment-invariant correlations. 
Extensive experiments on a self-built transmission test rig, a high-speed train, and a metro train bogie demonstrate the superiority, robustness, and interpretability of the proposed method. Visualization of relational subgraphs further provides intuitive interpretability support for distributed diagnostic decisions.

---

## 🧠 Framework Overview

### 🔧 Proposed Network Architecture

![EAGRR Framework](https://github.com/CHAOZHAO-1/EAGRR/blob/main/IMG1/F1.png)

---
## 📁 Project Structure

```text
EAGRR/
│
├── EAGRR.py
│   └── Main implementation of the proposed EAGRR framework
│
├── resnet18_1d.py
│   └── 1D CNN backbone network architecture
│
├── data_loader_1d.py
│   └── Data loading and preprocessing for 1D vibration signals
│
└── IMG1/
    └── Framework illustrations

```
---  

## 📊 Dataset
HUST Transmission System Dataset

🔗 Dataset Repository:
https://github.com/CHAOZHAO-1/HUSTTransmissionsystem-dataset

---  
@article{ZHAO2025130962,  
  title   = {Environment-Aware graph relational reasoning for interpretable and generalizable mechanical transmission system distributed fault diagnosis},  
  journal = {Expert Systems with Applications},  
  pages   = {130962},  
  year    = {2025},  
  issn    = {0957-4174},  
  doi     = {10.1016/j.eswa.2025.130962},  
  url     = {https://www.sciencedirect.com/science/article/pii/S0957417425045774},  
  author  = {Chao Zhao and Weiming Shen and Enrico Zio and Hui Ma},  
  keywords= {Mechanical transmission system, Domain generalization, Graph neural network, Distributed fault diagnosis, Multiple sensors}  
}  

