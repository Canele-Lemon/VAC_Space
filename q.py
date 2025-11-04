def _load_jacobian_bundle_npy(self):
    """
    estimate_jacobian.py 에서 만든 npy 번들을 로드.
    기대 구조:
      bundle["J"]   : (256,3,3)
      bundle["n"]   : (256,)
      bundle["cond"]: (256,)
    """
    jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_dense.npy')  # 파일명은 실제꺼로 수정
    if not os.path.exists(jac_path):
        logging.error(f"[Jacobian] npy 파일을 찾을 수 없습니다: {jac_path}")
        raise FileNotFoundError(f"Jacobian npy not found: {jac_path}")

    bundle = np.load(jac_path, allow_pickle=True).item()
    J = np.asarray(bundle["J"], dtype=np.float32)      # (256,3,3)
    n = np.asarray(bundle["n"], dtype=np.int32)        # (256,)
    cond = np.asarray(bundle["cond"], dtype=np.float32)

    self._jac_bundle = bundle
    self._J_dense = J
    self._J_n = n
    self._J_cond = cond

    logging.info(f"[Jacobian] dense J bundle loaded: {jac_path}, J.shape={J.shape}")