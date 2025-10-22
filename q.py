PS D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module> & C:/python310/python.exe "d:/00 업무/00 가상화기술/00 색시야각 보상 최적화/VAC algorithm/module/scripts/VACJacobianTrainer.py"
[Jacobian] using 1042 PKs

[Jacobian] Start training with 1042 PKs, 33 knots

=== Learn Jacobian for Y0-Gamma (vs High) ===
  └ X shape: (1046701, 210), y shape: (1046701,)
C:\python310\lib\site-packages\sklearn\linear_model\_ridge.py:213: LinAlgWarning: Ill-conditioned matrix (rcond=4.10912e-10): result may not be accurate.
  return linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T
  ⏱  Y0-Gamma done in 49.1 s

=== Learn Jacobian for Y0-Cx (vs High) ===
  └ X shape: (1067008, 210), y shape: (1067008,)
Traceback (most recent call last):
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\VACJacobianTrainer.py", line 319, in <module>
    main()
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\VACJacobianTrainer.py", line 316, in main
    train_jacobian_models(pk_list, out_path, knots_K=KNOTS)
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\VACJacobianTrainer.py", line 237, in train_jacobian_models
    model.fit(X, y)
  File "C:\python310\lib\site-packages\sklearn\base.py", line 1365, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\python310\lib\site-packages\sklearn\pipeline.py", line 663, in fit
    self._final_estimator.fit(Xt, y, **last_step_params["fit"])
  File "C:\python310\lib\site-packages\sklearn\base.py", line 1365, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\python310\lib\site-packages\sklearn\linear_model\_ridge.py", line 1248, in fit
    return super().fit(X, y, sample_weight=sample_weight)
  File "C:\python310\lib\site-packages\sklearn\linear_model\_ridge.py", line 990, in fit
    self.coef_, self.n_iter_, self.solver_ = _ridge_regression(
  File "C:\python310\lib\site-packages\sklearn\linear_model\_ridge.py", line 807, in _ridge_regression
    coef = _solve_svd(X, y, alpha, xp)
  File "C:\python310\lib\site-packages\sklearn\linear_model\_ridge.py", line 287, in _solve_svd
    U, s, Vt = xp.linalg.svd(X, full_matrices=False)
  File "C:\python310\lib\site-packages\sklearn\externals\array_api_compat\_internal.py", line 34, in wrapped_f
    return f(*args, xp=xp, **kwargs)
  File "C:\python310\lib\site-packages\sklearn\externals\array_api_compat\common\_linalg.py", line 76, in svd
    return SVDResult(*xp.linalg.svd(x, full_matrices=full_matrices, **kwargs))
  File "<__array_function__ internals>", line 200, in svd
  File "C:\python310\lib\site-packages\numpy\linalg\linalg.py", line 1642, in svd
    u, s, vh = gufunc(a, signature=signature, extobj=extobj)
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.67 GiB for an array with shape (1067008, 210) and data type float64
PS D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module>
