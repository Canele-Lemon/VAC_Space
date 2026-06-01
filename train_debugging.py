PS D:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project> & C:/python310/python.exe "d:/00 업무/00 가상화기술/25Y/00 색시야각 보상 최적화/VAC algorithm/VAC_Optimization_Project/src/modeling/train_model.py"
▶ Train with 1560 PKs
▶ Mapping file: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\data\vac_set_mapping.csv
 pk_start  pk_end  ref_pk  base_pk model_name panel_maker  frame_rate model_year                     memo
     4254    4566    4254     4255   50QNED85         INX         120        Y25    ref_pk=4565,4348,4336
     3943    4253    3943     3944   50QNED85     HKC(H2)         120        Y25              ref_pk=4163
     3631    3942    3631     3632     43UT80  CSOT(CSPI)          60        Y24         ref_pk=3931,3940
     3320    3630    3320     3321   43NANO80     HKC(H2)          60        Y25              ref_pk=3553
     3007    3319    3007     3008     50UB85         INX          60        Y26 ref_pk=3317/base_pk=3318
INFO:root:[VACDataset] Use_Flag='N' 이라 제외된 PK: [3317, 3318, 3553, 3931, 3940, 4163, 4336, 4348, 4565]
▶ Valid PKs after Use_Flag filtering: 1551
▶ Collected samples: 1551

=== Train Y0: dGamma ===
⏱️ [Y0-dGamma] Linear fit: 0.2s | MSE=0.003478 R²=0.223856
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  51.4s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  51.8s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  56.2s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.0min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.1min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.1min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.2min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.3min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.8min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.8min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  39.0s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  39.3s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.5s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  37.4s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.2min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  29.3s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  29.5s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.8min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.2min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.2min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.2min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.3min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.2min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  56.7s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.2min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  56.8s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.5min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.3min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  55.9s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.4min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time= 1.0min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time= 1.0min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time= 1.0min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.9min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.8min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.8min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time= 1.1min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time= 1.0min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time= 1.1min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.1min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.0min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.1min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.5min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.5min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  59.2s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  59.9s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  59.3s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.5min
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  32.4s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  31.4s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  31.0s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  45.3s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  44.9s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  44.3s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.2min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.1min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.1min
⏱️ [Y0-dGamma] RF(residual) search: 12.7 min
✅ [Y0-dGamma] RF best params: {'max_depth': 17, 'max_features': 0.5598033066958126, 'min_samples_leaf': 13, 'min_samples_split': 7, 'n_estimators': 133}
✅ [Y0-dGamma] RF best R² (CV): 0.946552
🏁 [Y0-dGamma] Hybrid — MSE:0.000174 R²:0.961234
📁 saved: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\modeling\hybrid_dGamma_model.pkl

=== Train Y0: dCx ===
⏱️ [Y0-dCx] Linear fit: 0.2s | MSE=0.000012 R²=0.369124
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  51.9s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  51.9s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  54.1s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.3min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.3min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.3min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.4min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.5min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 2.0min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  38.5s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.9s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  38.9s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 2.1min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  38.0s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.4min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.0s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.9s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.2min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.2min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.2min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 2.0min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.4min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.4min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  55.0s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  55.6s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.5min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.4min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  59.9s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.6min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.6min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time= 1.0min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time= 1.1min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 2.0min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  59.9s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 2.0min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 2.0min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time= 1.1min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time= 1.1min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time= 1.1min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.1min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.2min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.2min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.5min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.6min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  58.9s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.5min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  60.0s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  58.2s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.4s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  29.5s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  29.4s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  43.0s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  42.9s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  43.1s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.4min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.4min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.4min
⏱️ [Y0-dCx] RF(residual) search: 15.4 min
✅ [Y0-dCx] RF best params: {'max_depth': 17, 'max_features': 0.9722042458113105, 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 160}
✅ [Y0-dCx] RF best R² (CV): 0.922507
🏁 [Y0-dCx] Hybrid — MSE:0.000001 R²:0.953518
📁 saved: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\modeling\hybrid_dCx_model.pkl

=== Train Y0: dCy ===
⏱️ [Y0-dCy] Linear fit: 0.2s | MSE=0.000036 R²=0.425287
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  51.0s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  52.4s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  52.6s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.0min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.0min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 2.0min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.2min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.2min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  34.5s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.7min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  36.4s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.8min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  26.5s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  35.3s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 2.2min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.0s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  27.5s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.8min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.5min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.2min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.2min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.3min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time= 1.2min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  54.4s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  53.5s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  55.7s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.3min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.4min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  58.4s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  58.4s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.7min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  55.2s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.7min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.7min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  59.2s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  59.2s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  59.4s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.9min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.0min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 2.0min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.4min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.4min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  53.2s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  54.3s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.4min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  54.2s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.9s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.6s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.8s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  42.6s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  42.5s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  41.7s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.1min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.1min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 2.0min
⏱️ [Y0-dCy] RF(residual) search: 13.8 min
✅ [Y0-dCy] RF best params: {'max_depth': 17, 'max_features': 0.9722042458113105, 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 160}
✅ [Y0-dCy] RF best R² (CV): 0.935396
🏁 [Y0-dCy] Hybrid — MSE:0.000002 R²:0.961848
📁 saved: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\modeling\hybrid_dCy_model.pkl

=== Train Y1 ===
⏱️ [Y1-slope] Linear fit: 0.0s | MSE=0.019510 R²=0.302873
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=   3.0s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=   3.0s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=   3.0s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time=   5.3s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time=   5.3s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time=   5.4s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time=   6.5s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time=   6.6s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=   2.1s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=   2.0s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time=   4.9s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time=   5.0s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=   2.2s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=   2.5s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=   2.3s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time=   7.0s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=   2.3s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time=   5.5s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time=   4.2s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time=   4.1s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time=   4.1s
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time=   5.0s
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time=   4.9s
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time=   4.8s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=   3.4s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=   3.4s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=   3.4s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time=   7.2s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time=   7.1s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=   3.5s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=   3.4s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time=   7.2s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=   3.4s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=   3.5s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time=   5.3s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=   3.6s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=   3.6s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time=   5.3s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time=   5.3s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=   2.6s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=   2.6s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=   2.5s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time=   6.1s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time=   5.9s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time=   3.7s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time=   3.5s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time=   6.1s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=   3.0s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=   3.0s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time=   4.0s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=   3.4s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=   2.5s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=   2.5s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=   2.5s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time=   5.4s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time=   5.3s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time=   5.2s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=   3.1s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=   2.8s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=   2.3s
⏱️ [Y1-slope] RF(residual) search: 0.7 min
✅ [Y1-slope] RF best params: {'max_depth': 12, 'max_features': 0.5109418317515857, 'min_samples_leaf': 5, 'min_samples_split': 6, 'n_estimators': 143}
✅ [Y1-slope] RF best R² (CV): 0.972032
🏁 [Y1-slope] Hybrid — MSE:0.000653 R²:0.976667
📁 saved: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\modeling\hybrid_Y1_slope_model.pkl

=== Train Y2 (delta_uv) ===
⏱️ [Y2-delta_uv] Linear fit: 0.0s | MSE=0.000003 R²=0.963173
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=   2.0s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=   2.1s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=   2.1s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time=   3.2s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time=   3.2s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time=   3.2s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time=   3.8s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time=   4.0s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=   1.4s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=   1.3s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time=   2.8s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time=   2.8s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=   1.6s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=   1.9s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time=   3.2s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time=   4.5s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=   1.9s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=   1.9s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time=   2.8s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time=   2.7s
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time=   2.7s
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time=   3.0s
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time=   2.9s
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time=   2.8s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=   1.8s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=   1.9s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=   1.9s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time=   4.1s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time=   4.2s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=   2.1s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=   2.1s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time=   4.3s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=   2.2s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=   2.4s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time=   3.1s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time=   3.2s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=   2.4s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time=   3.2s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=   2.4s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=   1.5s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=   1.6s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time=   3.5s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=   1.6s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time=   3.5s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time=   2.1s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time=   2.1s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time=   3.7s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time=   2.4s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=   1.8s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=   1.9s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=   1.9s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time=   3.2s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=   2.1s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time=   3.2s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time=   3.3s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=   2.2s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=   2.0s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=   2.2s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=   2.0s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=   1.2s
⏱️ [Y2-delta_uv] RF(residual) search: 0.4 min
✅ [Y2-delta_uv] RF best params: {'max_depth': 17, 'max_features': 0.9722042458113105, 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 160}
✅ [Y2-delta_uv] RF best R² (CV): 0.944005
🏁 [Y2-delta_uv] Hybrid — MSE:0.000000 R²:0.998478
📁 saved: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\modeling\hybrid_Y2_delta_uv_model.pkl
