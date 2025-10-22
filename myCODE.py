    def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
        logging.info(f"[CORR] iteration {iter_idx} start")

        # 1) 현재 TV LUT (캐시) 확보
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.warning("[CORR] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
            return None
        vac_dict = self._vac_dict_cache # 표준 키 dict

        # 2) 4096→256 다운샘플 (High만 수정, Low 고정)
        #    원래 키 → 표준 LUT 키로 꺼내 계산
        vac_lut = {
            "R_Low":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "R_High": np.asarray(vac_dict["RchannelHigh"], dtype=np.float32),
            "G_Low":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "G_High": np.asarray(vac_dict["GchannelHigh"], dtype=np.float32),
            "B_Low":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "B_High": np.asarray(vac_dict["BchannelHigh"], dtype=np.float32),
        }
        high_256 = {ch: self._down4096_to_256(vac_lut[ch]) for ch in ['R_High','G_High','B_High']}
        # low_256  = {ch: self._down4096_to_256(vac_lut[ch]) for ch in ['R_Low','G_Low','B_Low']}

        # 3) Δ 목표(white/main 기준): OFF vs ON 차이를 256 길이로 구성
        #    Gamma: 1..254 유효, Cx/Cy: 0..255
        d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
        # d_targets: {"Gamma":(256,), "Cx":(256,), "Cy":(256,)}

        # 4) 결합 선형계: [wG*A_Gamma; wC*A_Cx; wC*A_Cy] Δh = - [wG*ΔGamma; wC*ΔCx; wC*ΔCy]
        wG, wC = 1.0, 1.0
        A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
        b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)

        # 유효치 마스크(특히 gamma의 NaN)
        mask = np.isfinite(b_cat)
        A_use = A_cat[mask, :]
        b_use = b_cat[mask]

        # 5) 리지 해(Δh) 구하기 (3K-dim: [Rknots, Gknots, Bknots])
        #    (A^T A + λI) Δh = A^T b
        ATA = A_use.T @ A_use
        rhs = A_use.T @ b_use
        ATA[np.diag_indices_from(ATA)] += lambda_ridge
        delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

        # 6) Δcurve = Phi * Δh_channel 로 256-포인트 보정곡선 만들고 High에 적용
        K    = len(self._jac_artifacts["knots"])
        dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
        Phi  = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)
        corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

        high_256_new = {
            "R_High": (high_256["R_High"] + corr_R).astype(np.float32),
            "G_High": (high_256["G_High"] + corr_G).astype(np.float32),
            "B_High": (high_256["B_High"] + corr_B).astype(np.float32),
        }

        # 7) High 경계/단조/클램프 → 12bit 업샘플 & Low는 유지하여 "표준 dict 구성"
        for ch in high_256_new:
            self._enforce_monotone(high_256_new[ch])
            high_256_new[ch] = np.clip(high_256_new[ch], 0, 4095)

        new_lut_tvkeys = {
            "RchannelLow":  np.asarray(self._vac_dict_cache["RchannelLow"], dtype=np.float32),
            "GchannelLow":  np.asarray(self._vac_dict_cache["GchannelLow"], dtype=np.float32),
            "BchannelLow":  np.asarray(self._vac_dict_cache["BchannelLow"], dtype=np.float32),
            "RchannelHigh": self._up256_to_4096(high_256_new["R_High"]),
            "GchannelHigh": self._up256_to_4096(high_256_new["G_High"]),
            "BchannelHigh": self._up256_to_4096(high_256_new["B_High"]),
        }

        vac_write_json = self.build_vacparam_std_format(self._vac_dict_cache, new_lut_tvkeys)

        def _after_write(ok, msg):
            logging.info(f"[VAC Write] {msg}")
            if not ok:
                return
            # 쓰기 성공 → 재읽기
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            if not vac_dict_after:
                logging.error("보정 후 VAC 재읽기 실패")
                return
            # ✅ 여기서 캐시 갱신 (성공 케이스에만)
            self._vac_dict_cache = vac_dict_after
            # 차트용 변환 후 표시
            lut_dict_plot = {k.replace("channel","_"): v
                            for k, v in vac_dict_after.items() if "channel" in k}
            self._update_lut_chart_and_table(lut_dict_plot)
            # 다음 측정 세션 시작 등...

        # TV에 적용
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)

여기서 1차 보정 후에 아래 에러가 발생합니다:
2025-10-22 10:31:50,146 - INFO - subpage_vacspace.py:990 - [CORR] iteration 1 start
2025-10-22 10:31:50,186 - DEBUG - subpage_vacspace.py:5816 - Sending command: s
2025-10-22 10:31:50,395 - DEBUG - subpage_vacspace.py:5847 - Output: s
/bin/sh: s: not found
/ #
/ #
2025-10-22 10:31:50,395 - DEBUG - subpage_vacspace.py:5816 - Sending command: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || echo 'not_exists'
2025-10-22 10:31:50,608 - DEBUG - subpage_vacspace.py:5847 - Output: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || ech
o 'not_exists'
exists
/ #
/ #
2025-10-22 10:31:50,609 - DEBUG - subpage_vacspace.py:5816 - Sending command: cp /etc/panelcontroller/db/vac/vac_INX_50_60hz.json /mnt/lg/cmn_data/panelcontroller/db/vac_debug
2025-10-22 10:31:50,822 - DEBUG - subpage_vacspace.py:5847 - Output: cp /etc/panelcontroller/db/vac/vac_INX_50_60hz.json /mnt/lg/cmn_data/panelco
ntroller/db/vac_debug
/ #
/ #
2025-10-22 10:32:03,954 - DEBUG - subpage_vacspace.py:5845 - Output (truncated): cat > /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
{
"DRV_valc_major_ctrl"   :       [       0,      1       ],
"DRV_valc_pattern_ctrl_0"       :       [       11,     1       ],
"DRV_valc_pattern_ctrl_1"       :       [       [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ]               ],
"DRV_valc_sat_ctrl"     :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0       ],
"DRV_valc_hpf_ctrl_0"   :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      7       ],
"DRV_valc_hpf_ctrl_1"   :               1,
"RchannelLow"   :       [       0,      0,      0,      0,      1,      1,      1,      1,      1,      1,      1,      1,      2,      2,      2,      2,
                        2,      2,      2,      2,      3,      3,      3,      3,      3,      3,      3,      3,      4,      4,      4,      4,
                        4,      4,      4,      4,      5,      5,      5,      5,      5,      5,      5,      5,      6,      6,      6,      6,
                        6,      6,      6,      6,      7,
2025-10-22 10:32:03,956 - DEBUG - subpage_vacspace.py:5816 - Sending command: restart panelcontroller
2025-10-22 10:32:04,167 - DEBUG - subpage_vacspace.py:5847 - Output: restart panelcontroller
restart panelcontroller
/ #
/ #
2025-10-22 10:32:04,168 - DEBUG - subpage_vacspace.py:5816 - Sending command: exit
2025-10-22 10:32:04,379 - DEBUG - subpage_vacspace.py:5847 - Output: exit
exit shell mode

009669.622263:teminalmanager-h:help] 0x0000000a
009669.622293:teminalmanager-h:help] 0x0000000a
009669.622319:teminalmanager-h:help] 0x0000000a
2025-10-22 10:32:04,380 - INFO - subpage_vacspace.py:1062 - [VAC Write] VAC data written to /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
2025-10-22 10:32:04,382 - DEBUG - subpage_vacspace.py:5816 - Sending command: s
2025-10-22 10:32:04,592 - DEBUG - subpage_vacspace.py:5847 - Output: 
009669.836364:teminalmanager-h:help] 0x00000073
start shell mode.
/ #
2025-10-22 10:32:04,593 - DEBUG - subpage_vacspace.py:5816 - Sending command: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || echo 'not_exists'
2025-10-22 10:32:04,806 - DEBUG - subpage_vacspace.py:5847 - Output: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || ech
o 'not_exists'
exists
/ #
/ #
2025-10-22 10:32:04,807 - DEBUG - subpage_vacspace.py:5816 - Sending command: cat /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
2025-10-22 10:32:17,691 - DEBUG - subpage_vacspace.py:5845 - Output (truncated): cat /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
{
"DRV_valc_major_ctrl"   :       [       0,      1       ],
"DRV_valc_pattern_ctrl_0"       :       [       11,     1       ],
"DRV_valc_pattern_ctrl_1"       :       [       [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ]               ],
"DRV_valc_sat_ctrl"     :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0       ],
"DRV_valc_hpf_ctrl_0"   :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      7       ],
"DRV_valc_hpf_ctrl_1"   :               1,
"RchannelLow"   :       [       0,      0,      0,      0,      1,      1,      1,      1,      1,      1,      1,      1,      2,      2,      2,      2,
                        2,      2,      2,      2,      3,      3,      3,      3,      3,      3,      3,      3,      4,      4,      4,      4,
                        4,      4,      4,      4,      5,      5,      5,      5,      5,      5,      5,      5,      6,      6,      6,      6,
                        6,      6,      6,      6,      7,      7,
2025-10-22 10:32:17,967 - ERROR - subpage_vacspace.py:1567 - 'LUTChart' object has no attribute 'lines'
Traceback (most recent call last):
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1552, in _update_lut_chart_and_table
    if label not in chart.lines:
AttributeError: 'LUTChart' object has no attribute 'lines'
