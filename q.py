# debug_check_y.py (예시 파일 이름)

import numpy as np

from src.data_preparation.prepare_output import VACOutputBuilder

def main():
    ref_pk    = 2744   # base LUT pk
    base_pk   = 2744   # Base (엑셀의 Base에 해당)
    target_pk = 2779   # G+50 or B+50 같은 스윕 PK
    gray      = 128    # 보고 싶은 gray
    pattern   = 'W'    # White 패턴 기준

    print(f"ref_pk = {ref_pk}, base_pk = {base_pk}, target_pk = {target_pk}, gray = {gray}, pattern = {pattern}")
    print("===================================================================")

    # 1) 절대값 Y0_abs (각 PK의 절대 Cx/Cy/Gamma)
    base_builder   = VACOutputBuilder(pk=base_pk,   ref_pk=ref_pk)
    target_builder = VACOutputBuilder(pk=target_pk, ref_pk=ref_pk)

    y_abs_base   = base_builder.compute_Y0_struct_abs()   # { 'W': {'Gamma','Cx','Cy'}, ... }
    y_abs_target = target_builder.compute_Y0_struct_abs()

    cx_base  = float(y_abs_base[pattern]['Cx'][gray])
    cy_base  = float(y_abs_base[pattern]['Cy'][gray])
    gam_base = float(y_abs_base[pattern]['Gamma'][gray])

    cx_tgt  = float(y_abs_target[pattern]['Cx'][gray])
    cy_tgt  = float(y_abs_target[pattern]['Cy'][gray])
    gam_tgt = float(y_abs_target[pattern]['Gamma'][gray])

    print(f"[ABS] Base PK={base_pk}, gray={gray}")
    print(f"  Cx = {cx_base:.6f}, Cy = {cy_base:.6f}, Gamma = {gam_base:.6f}")
    print(f"[ABS] Target PK={target_pk}, gray={gray}")
    print(f"  Cx = {cx_tgt:.6f}, Cy = {cy_tgt:.6f}, Gamma = {gam_tgt:.6f}")

    dCx_manual  = cx_tgt  - cx_base
    dCy_manual  = cy_tgt  - cy_base
    dGam_manual = gam_tgt - gam_base
    print(f"[MANUAL diff] (Target - Base)")
    print(f"  dCx = {dCx_manual:+.6f}, dCy = {dCy_manual:+.6f}, dGamma = {dGam_manual:+.6f}")
    print("-------------------------------------------------------------------")

    # 2) prepare_Y()가 계산하는 dY (ref_pk=2744 기준)
    #   → compute_Y0_struct() 내부에서 pk vs ref_pk 차이를 이미 계산해 줌
    target_builder_rel = VACOutputBuilder(pk=target_pk, ref_pk=ref_pk)
    Y_target = target_builder_rel.prepare_Y(
        add_y0=True, add_y1=False, add_y2=False,
        y0_patterns=(pattern,)
    )
    y0_tgt = Y_target["Y0"][pattern]   # {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)}

    dCx_ref  = float(y0_tgt["dCx"][gray])
    dCy_ref  = float(y0_tgt["dCy"][gray])
    dGam_ref = float(y0_tgt["dGamma"][gray])

    print(f"[prepare_Y dY] PK={target_pk} vs ref_pk={ref_pk}, pattern={pattern}, gray={gray}")
    print(f"  dCx = {dCx_ref:+.6f}, dCy = {dCy_ref:+.6f}, dGamma = {dGam_ref:+.6f}")

    # 3) manual diff와 prepare_Y 결과가 같은지 체크
    print("-------------------------------------------------------------------")
    print("[CHECK] manual (Target - Base) vs prepare_Y dY (Target vs Ref)")

    print(f"  dCx: manual={dCx_manual:+.6f}, prepare_Y={dCx_ref:+.6f}, diff={dCx_ref - dCx_manual:+.6e}")
    print(f"  dCy: manual={dCy_manual:+.6f}, prepare_Y={dCy_ref:+.6f}, diff={dCy_ref - dCy_manual:+.6e}")
    print(f"  dG : manual={dGam_manual:+.6f}, prepare_Y={dGam_ref:+.6f}, diff={dGam_ref - dGam_manual:+.6e}")

if __name__ == "__main__":
    main()