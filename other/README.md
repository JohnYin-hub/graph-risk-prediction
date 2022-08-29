# icdm 2022 baseline

We implement [RCCN](https://arxiv.org/abs/1703.06103) baseline on [ICDM 2022 : Risk Commodities Detection on Large-Scale E-commerce Graphs](https://tianchi.aliyun.com/competition/entrance/531976/introduction). Our implementation include three popular GNN platforms: [DGL](https://github.com/dmlc/dgl), [PyG](https://github.com/pyg-team/pytorch_geometric) and [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN). You can use it as a reference for data processing, model training and inference.

## Environment Setup

- **Option1:** Run on OpenI Cloud platform.
  - We provide an image with pytorch, dgl and pyg installed. You can directly create an instance from that. Also, you can use dataset provided on OpenI.
- **Option2:** Run locally.
  - Download the dataset and config the environment locally.

## Experiment Result

| AP       | session1 | session2 |
|----------|----------|----------|
| pyg      | 0.9174   | 0.8931   |               
| dgl      | 0.8751   | 0.8417   |               
| openhgnn | 0.8658   | 0.8387   |               
