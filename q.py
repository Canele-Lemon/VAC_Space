PS D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module> & C:/python310/python.exe "d:/00 업무/00 가상화기술/00 색시야각 보상 최적화/VAC algorithm/module/scripts/VACJacobianTrainer.py"
[Jacobian] using 1042 PKs

[Jacobian-DELTA] Start training with 1042 PKs, 33 knots

=== Learn Jacobian for dGamma (ΔLUT High+Low) ===
  └ X shape: (263401, 210), y shape: (263401,)
  ⏱  dGamma done in 10.0 s

=== Learn Jacobian for dCx (ΔLUT High+Low) ===
  └ X shape: (266752, 210), y shape: (266752,)
  ⏱  dCx done in 10.1 s

=== Learn Jacobian for dCy (ΔLUT High+Low) ===
  └ X shape: (266752, 210), y shape: (266752,)
  ⏱  dCy done in 10.2 s

✅ All components trained in 10.0 min
📁 saved Jacobian model: d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\jacobian_INX_60_K33.pkl
[DEBUG] A_dGamma shape: (256, 198)
        first row few elems: [ 0.01602387  0.          0.          0.          0.          0.
  0.         -0.          0.         -0.        ]
