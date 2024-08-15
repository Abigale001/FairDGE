# FairDGE

This is the offical pytorch implementation of KDD24 paper "[Toward Structure Fairness in Dynamic Graph Embedding: A Trend-aware Dual Debiasing Approach](https://arxiv.org/pdf/2406.13201)". 

## Abstract
Recent studies successfully learned static graph embeddings that are structurally fair by preventing the effectiveness disparity of high- and low-degree vertex groups in downstream graph mining tasks. However, achieving structure fairness in dynamic graph embedding remains an open problem. Neglecting degree changes in dynamic graphs will significantly impair embedding effectiveness without notably improving structure fairness. This is because the embedding performance of high-degree and low-to-high-degree vertices will significantly drop close to the generally poorer embedding performance of most slightly changed vertices in the longtail part of the power-law distribution. We first identify biased structural evolutions in a dynamic graph based on the evolving trend of vertex degree and then propose FairDGE, the first structurally Fair Dynamic Graph Embedding algorithm. FairDGE learns biased structural evolutions by jointly embedding the connection changes among vertices and the long-short-term evolutionary trend of vertex degrees. Furthermore, a novel dual debiasing approach is devised to encode fair embeddings contrastively, customizing debiasing strategies for different biased structural evolutions. This innovative debiasing strategy breaks the effectiveness bottleneck of embeddings without notable fairness loss. Extensive experiments demonstrate that FairDGE achieves simultaneous improvement in the effectiveness and fairness of embeddings.



## Environment
- Python 3.8.16
- pytorch 2.0.1
- matplotlib 3.7.1
- networkx 2.8.4
- numpy 1.24.3
- pandas 1.5.3
- scikit-learn 1.2.2
  
See environment.yml

## Quick Start
`python main.py --cuda 0 --data_name MovieLens`

The first running may take a long time for data preparation.

## Citation
> Li Y, Yang Y, Cao J, et al. Toward Structure Fairness in Dynamic Graph Embedding: A Trend-aware Dual Debiasing Approach[J]. arXiv preprint arXiv:2406.13201, 2024.
