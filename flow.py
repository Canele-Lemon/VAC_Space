start_VAC_optimization
  ├─ [JAC] _load_jacobian_artifacts() → artifacts
  │    └─ _build_A_from_artifacts("Gamma") → A_Gamma (256 × 3K)
  │    └─ _build_A_from_artifacts("Cx")    → A_Cx
  │    └─ _build_A_from_artifacts("Cy")    → A_Cy
  │
  ├─ [INIT] 측정 버퍼 초기화: _off_store / _on_store
  ├─ [TV] _set_vac_active(False)   # VAC OFF 보장
  └─ _run_off_baseline_then_on()
       └─ start_viewing_angle_session(profile=OFF, gamma_lines=OFF용, ...)
            ├─ (세션 루프)
            │    ├─ γ 단계: patterns×gray_levels 순회
            │    │    ├─ _trigger_gamma_pair() → MeasureThread(main/sub)
            │    │    └─ _consume_gamma_pair(): store['gamma'] 갱신
            │    │         ├─ GammaChart 실시간 업데이트(메인/서브 축)
            │    │         └─ (white/main) 테이블 Lv/Cx/Cy 기록
            │    └─ 색편차 단계: cs_patterns 순회
            │         ├─ _trigger_colorshift_pair() → MeasureThread(main/sub)
            │         └─ _consume_colorshift_pair(): store['colorshift'] 갱신
            │              └─ CIE1976ChromaticityDiagram 점 갱신
            └─ _finalize_session()
                 └─ white/main Lv → _compute_gamma_series() → Γ 벡터 테이블 기록
                 └─ on_done 콜백: _after_off(store_off)
                      ├─ self._off_store ← store_off
                      ├─ [TV] _set_vac_active(True) (필요 시)
                      └─ _apply_vac_from_db_and_measure_on()
                           ├─ [DB] _fetch_vac_by_model(panel, fr)
                           │    └─ (vac_info_pk, vac_version, vac_data)
                           ├─ [TV] _write_vac_to_tv(vac_data)
                           ├─ [TV] _read_vac_from_tv(on_finished=_after_read)
                           │    └─ _after_read(vac_dict)
                           │         ├─ self._vac_dict_cache ← vac_dict   # 원본 캐시
                           │         ├─ (플롯용) lut_dict_plot 생성
                           │         ├─ _update_lut_chart_and_table(lut_dict_plot)
                           │         └─ start_viewing_angle_session(profile=ON, gamma_lines=ON용, ...)
                           │              └─ (OFF 세션과 동일 루틴)
                           │              └─ _finalize_session()
                           │                   └─ on_done 콜백: _after_on(store_on)
                           │                        ├─ self._on_store ← store_on
                           │                        ├─ [SPEC] _check_spec_pass(_off_store, _on_store)
                           │                        │     ├─ 통과 → ✅ 종료
                           │                        │     └─ 미통과 → 보정 진입
                           │                        └─ _run_one_correction_cycle(iter=1)
                           │                             └─ _run_correction_iteration(iter_idx=1, max_iters=2, λ=1e-3)
                           │                                  ├─ [LUT] self._vac_dict_cache에서 4096 High/Low 추출
                           │                                  ├─ 4096→256 다운샘플(High만 대상)
                           │                                  ├─ [Δ타깃] _build_delta_targets_from_stores(OFF, ON)
                           │                                  │     → ΔΓ, ΔCx, ΔCy (white/main 기준)
                           │                                  ├─ [선형계] [wG·AΓ; wC·ACx; wC·ACy] Δh = -[wG·ΔΓ; wC·ΔCx; wC·ΔCy]
                           │                                  │     └─ 리지해 Δh 산출
                           │                                  ├─ Φ·Δh → 채널별 보정커브(256) 생성
                           │                                  ├─ 단조/클램프(_enforce_monotone) → 256→4096 업샘플(High만 교체)
                           │                                  ├─ build_vacparam_std_format(원본dict, new_highs)
                           │                                  ├─ [TV] _write_vac_to_tv(vac_write_json)
                           │                                  ├─ [TV] _read_vac_from_tv(_after_read_back)
                           │                                  │     └─ self._vac_dict_cache 갱신
                           │                                  │     └─ _update_lut_chart_and_table() 갱신
                           │                                  └─ (재측정) start_viewing_angle_session(profile=CORR_i, ...)
                           │                                        └─ _finalize_session()
                           │                                             └─ _after_corr(store_corr_i)
                           │                                                  ├─ [SPEC] 재검증: 통과 → ✅ 종료
                           │                                                  └─ (iter < max) 다음 사이클 반복
                           └─ (예외/에러) 로그 출력 후 안전 종료