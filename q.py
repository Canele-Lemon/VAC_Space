    def _load_jacobian_bundle_npy(self):
        """
        bundle["J"]   : (256,3,3)
        bundle["n"]   : (256,)
        bundle["cond"]: (256,)
        """
        if hasattr(self, "_jac_bundle") and self._jac_bundle is not None:
            return
        
        try:
            jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_bundle_ref2744_lam0.001_dw900.0_gs30.0_20251110_105631.npy')
            if not os.path.exists(jac_path):
                raise FileNotFoundError(f"Jacobian npy not found: {jac_path}")

            bundle = np.load(jac_path, allow_pickle=True).item()
            J = np.asarray(bundle["J"], dtype=np.float32) # (256, 3, 3)
            n = np.asarray(bundle["n"], dtype=np.int32)   # (256,)
            cond = np.asarray(bundle["cond"], dtype=np.float32)

            self._jac_bundle = bundle
            self._J_dense = J
            self._J_n = n
            self._J_cond = cond

            logging.info(f"[Jacobian] dense J bundle loaded: {jac_path}, J.shape={J.shape}")

        except Exception:
            logging.exception("[Jacobian] Jacobian load failed")
            raise
