def debug_print_XY_at_grays(pk_list, ref_pk, grays=(0, 1, 64, 128, 192, 254, 255), max_rows_per_gray=5):
    """
    지정한 gray들에 대해 X/Y 행을 그대로 출력한다.
    - max_rows_per_gray: 각 gray에서 최대 몇 개 행 출력할지 (너무 많이 찍히는 것 방지)
    """
    X, Y0, groups, idx_gray, ds = build_white_X_Y0(pk_list, ref_pk)

    gray_norm = X[:, idx_gray]
    gray_idx  = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    print("\n================ DEBUG: X/Y rows at selected grays ================")
    print(f"ref_pk={ref_pk}, pk_list_size={len(pk_list)}, total_rows={len(X)}")
    print(f"idx_gray={idx_gray}")
    print(f"target grays={list(grays)}")
    print("-------------------------------------------------------------------")

    for g in grays:
        m = (gray_idx == int(g))
        n = int(m.sum())
        print(f"\n[gray={g}] rows={n}")

        if n == 0:
            continue

        # 너무 많이 출력되는 것 방지
        idxs = np.where(m)[0][:max_rows_per_gray]

        for i in idxs:
            pk = int(groups[i])

            x_row = X[i]
            y_row = Y0[i]

            # 보기 편하게: ΔRGB(앞 3개) / 나머지 feature는 길이가 길 수 있으니 일부만
            dR, dG, dB = float(x_row[0]), float(x_row[1]), float(x_row[2])
            dCx, dCy, dGam = float(y_row[0]), float(y_row[1]), float(y_row[2])

            print(f"  - row={i}, pk={pk}")
            print(f"    X[0:3] (dR,dG,dB) = ({dR:+.3f}, {dG:+.3f}, {dB:+.3f})")
            print(f"    X(full) = {np.array2string(x_row, precision=4, floatmode='fixed')}")
            print(f"    Y (dCx,dCy,dGamma) = ({dCx:+.6f}, {dCy:+.6f}, {dGam:+.6f})")

    print("\n===================================================================\n")
    
def main():
    ref_pk = 3008
    pk_list = [3157]

    debug_print_XY_at_grays(
        pk_list=pk_list,
        ref_pk=ref_pk,
        grays=(0, 1, 64, 128, 192, 254, 255),
        max_rows_per_gray=5
    )