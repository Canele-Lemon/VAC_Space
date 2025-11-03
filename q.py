if __name__ == "__main__":
    target_pk = 2635
    reference_pk = 2582

    builder = VACOutputBuilder(pk=target_pk, reference_pk=reference_pk)
    Y = builder.prepare_Y(y1_patterns=('W',))  # Y0(dGamma/dCx/dCy), Y1, Y2 계산

    # 콘솔에서 일부만 미리보기: 0,1,32,128,255
    for p in ('W','R','G','B'):
        dG = Y['Y0'][p]['dGamma']; dCx = Y['Y0'][p]['dCx']; dCy = Y['Y0'][p]['dCy']
        print(f"\n[PTN {p}] (target {target_pk} - ref {reference_pk})")
        for g in (0,1,32,128,255):
            vG = None if not np.isfinite(dG[g]) else float(dG[g])
            vCx = None if not np.isfinite(dCx[g]) else float(dCx[g])
            vCy = None if not np.isfinite(dCy[g]) else float(dCy[g])
            print(f"  gray {g:3d}: dGamma={vG}, dCx={vCx}, dCy={vCy}")