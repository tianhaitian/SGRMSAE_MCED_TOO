README for the codes of 
SGR-MSAE: Single-graph-regularized supervised autoencoder for multi-cancer early detection and tissue-of-origin prediction

1. Download the publicly available cfDNA dataset from 
https://doi.org/10.6084/m9.figshare.24145392.v1
Nguyen, V.T.C., Nguyen, T.H., Doan, N.N.T., Pham, T.M.Q., Nguyen, G.T.H., Nguyen, T.D., Tran, T.T.T., Vo, D.L., Phan, T.H., Jasmine, T.X., et al.: Mul-
timodal analysis of methylomics and fragmentomics in plasma cell-free dna for multi-cancer early detection and localization. Elife 12, 89083 (2023)

2. Comparison with conventional and deep-learning benchmarks (benchmark of 20 methods).
2.1. 01_early_fusion_10seeds.py (for MCED and TOO, with LR, RF, SVM, XGB)
2.2. 02_PCA_fusion_10seeds.py 
2.3. 03_Tucker_fusion_10seeds.py
2.4. 04_deep_learning_10seeds.py
2.5. 05_SGRMSAE_MCED_10seeds.py
2.6. 05_SGRMSAE_TOO_10seeds.py
Results are presented in Figure 2, Figure S1 and S2
