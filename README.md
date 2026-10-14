SGR-MSAE: Single-graph-regularized supervised autoencoder for multi-cancer early detection and tissue-of-origin prediction
Juntao Li, Lubin Cui, Shuxia Feng, Tianhai Tian. (submitted)

1. Download the publicly available cfDNA dataset from 

https://doi.org/10.6084/m9.figshare.24145392.v1

Nguyen, V.T.C., Nguyen, T.H., Doan, N.N.T., Pham, T.M.Q., Nguyen, G.T.H., Nguyen, T.D., Tran, T.T.T., Vo, D.L., Phan, T.H., Jasmine, T.X., et al.: Mul-
timodal analysis of methylomics and fragmentomics in plasma cell-free dna for multi-cancer early detection and localization. Elife 12, 89083 (2023)

2. Comparison with conventional and deep-learning benchmarks (benchmark of 20 methods).

In Benchmark directory

2.1. 01_early_fusion_10seeds.py (for MCED and TOO, with LR, RF, SVM, XGB)

2.2. 02_PCA_fusion_10seeds.py 

2.3. 03_Tucker_fusion_10seeds.py

2.4. 04_deep_learning_10seeds.py

2.5. 05_SGRMSAE_MCED_10seeds.py

2.6. 05_SGRMSAE_TOO_10seeds.py

Results are presented in Figure 2, Figure S1 and S2

4. Contribution of patient-similarity graph regularization
Test 1. Compare SGR-MSAE without network, with random network, with real network (SGR-MSAE). Use code in Benchmark
In Benchmark directory
2.5. 05_SGRMSAE_MCED_10seeds.py 
2.6. 05_SGRMSAE_TOO_10seeds.py 
include these three network types. 
Results are reported in Table 1.
Test 2. test the combination of a global patient-similarity graph with module-specific graphs
In HybridgraphBeta directory
3.1. PCA_AE_HybridBeta_MCED.py
3.2. PCA_AE_HybridBeta_TOO.py
Results are presented in Figure S3. 

6. Sensitivity to module-wise PCA dimensionality
For approach A, a fixed target of 24 PCs per module, use code in Benchmark
In Benchmark directory
2.5. 05_SGRMSAE_MCED_10seeds.py 
2.6. 05_SGRMSAE_TOO_10seeds.py
For approach B, the minimum number required to explain 95\% of the variance within each module, use
In PCAdimensionality directory
3.1. PPCA_AE_Graph_MCED_var95.py
3.2. PCA_AE_Graph_TOO_var95.py
For approach C, the larger of 24 PCs and the number required to explain 95\% of the variance, use
3.3. PCA_AE_Graph_MCED_max24var95.py
3.4. PCA_AE_Graph_TOO_max24var95.py.py
Results are presented in Table 2 and Table S2. 

8. Predictive contributions of individual and combined cfDNA modules
Use one module only
In IndivualModule directory
5.1. MCED_add_remove_one.py
5.2. TOO_add_remove_one.py
Use two modules
5.3. MCED_ADD_two_module.py
5.4. TOO_ADD_two_module.py
Use three modules
5.5. MCED_ADD_three_module.py
5.6. TOO_ADD_three_module.py
Results are presented in Figures 3 and S4.

10. Effect of removing individual and combined cfDNA modules
Remove one module. Use the same code 5.1. and 5.2.
In IndivualModule directory
Remove two modules
6.1. MCED_REMOVE_two_module.py
6.2. TOO_REMOVE_two_module.py
Remove three modules
6.3. MCED_REMOVE_three_module.py
6.4. TOO_REMOVE_three_module.py
Results are presented in Figures 4 and 5. 

12. Performance on the fixed discovery-to-validation cohort
This study benchmarks 20 methods. Similar to part 2, each code has 4 methods
In discovery-to-valuation directory
7.1. 01_EarlyFusion4_heldout.py
7.2. 02_PCA4_heldout.py
7.3. 03_Tucker4_heldout.py
7.4. 04_DL4_heldout.py
7.5. 05_PCA_AE_GlobalGraph4_heldout.py
Results are presented in Figures 6.



