4. Sensitivity to module-wise PCA dimensionality
For approach A, a fixed target of 24 PCs per module, use code in Benchmark
2.5. 05_SGRMSAE_MCED_10seeds.py 
2.6. 05_SGRMSAE_TOO_10seeds.py
For approach B, the minimum number required to explain 95\% of the variance within each module, use 
3.1. PPCA_AE_Graph_MCED_var95.py
3.2. PCA_AE_Graph_TOO_var95.py
For approach C, the larger of 24 PCs and the number required to explain 95\% of the variance, use
3.3. PCA_AE_Graph_MCED_max24var95.py
3.4. PCA_AE_Graph_TOO_max24var95.py.py
Results are presented in Table 2 and Table S2. 
