def get_pk_list_by_panel_frame(self, panel_maker: str, frame_rate: int | float):
    """
    panel_maker + frame_rate 기준으로 해당되는 set의 PK 리스트 반환
    """
    matched = self.df[
        (self.df["panel_maker"].astype(str) == str(panel_maker)) &
        (self.df["frame_rate"].astype(float) == float(frame_rate))
    ]

    if matched.empty:
        raise KeyError(
            f"[VACSetMapping] No set found for panel_maker={panel_maker}, "
            f"frame_rate={frame_rate}"
        )

    pk_list = []
    for _, row in matched.iterrows():
        pk_list.extend(range(int(row["pk_start"]), int(row["pk_end"]) + 1))

    return pk_list


def iter_panel_frame_groups(self):
    """
    panel_maker + frame_rate 기준 group 정보 반환
    """
    group_cols = ["panel_maker", "frame_rate"]

    for (panel_maker, frame_rate), sub in self.df.groupby(group_cols):
        pk_list = []
        base_pks = []
        ref_pks = []
        model_names = []

        for _, row in sub.iterrows():
            pk_list.extend(range(int(row["pk_start"]), int(row["pk_end"]) + 1))
            base_pks.append(int(row["base_pk"]))
            ref_pks.append(int(row["ref_pk"]))
            model_names.append(str(row["model_name"]))

        yield {
            "panel_maker": str(panel_maker),
            "frame_rate": float(frame_rate),
            "pk_list": pk_list,
            "base_pks": base_pks,
            "ref_pks": ref_pks,
            "model_names": model_names,
        }
        
        
def estimate_jacobians_per_gray(
    pk_list,
    set_mapping,
    lam=1e-3,
    delta_window=None,
    gauss_sigma=None,
    min_samples=3,
):
    X, Y0, groups, idx_gray, ds = build_white_X_Y0(
        pk_list=pk_list,
        set_mapping=set_mapping,
    )
    
def build_white_X_Y0(pk_list, set_mapping):
    ds = VACDataset(
        pk_list=pk_list,
        set_mapping=set_mapping,
        drop_use_flag_N=True,
        reference_mode="base",
    )

    X_cx, y_cx, g_cx = ds._build_XY0_for_jacobian_g(component='dCx')
    X_cy, y_cy, g_cy = ds._build_XY0_for_jacobian_g(component='dCy')
    X_ga, y_ga, g_ga = ds._build_XY0_for_jacobian_g(component='dGamma')

    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])

    # 현재 feature 구조:
    # [dR_H, dG_H, dB_H] + panel_onehot(K) + frame_rate + gray_norm + LUT_j
    idx_gray = 3 + K + 1

    ...
    
==
        
=

def safe_tag(text):
    return (
        str(text)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def make_group_paths(panel_maker, frame_rate, lam, delta_window, gauss_sigma):
    os.makedirs("artifacts", exist_ok=True)

    maker_tag = safe_tag(panel_maker)
    frame_tag = f"{int(float(frame_rate))}Hz"

    tag = f"{maker_tag}_{frame_tag}_base_lam{lam}"

    if delta_window is not None:
        tag += f"_dw{float(delta_window)}"
    if gauss_sigma is not None:
        tag += f"_gs{float(gauss_sigma)}"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv = os.path.join(
        "artifacts",
        f"jacobians_white_high_{tag}_{ts}.csv"
    )
    out_npy = os.path.join(
        "artifacts",
        f"jacobian_bundle_{tag}_{ts}.npy"
    )

    return out_csv, out_npy
    
def main():
    start_time = time.time()

    mapping = VACSetMapping()

    lam = 1e-3
    delta_window = None
    gauss_sigma = None
    min_samples = 3

    for group in mapping.iter_panel_frame_groups():
        panel_maker = group["panel_maker"]
        frame_rate = group["frame_rate"]
        pk_list = group["pk_list"]

        print("\n" + "=" * 100)
        print(f"[JACOBIAN GROUP] panel_maker={panel_maker}, frame_rate={frame_rate}")
        print(f"model_names={group['model_names']}")
        print(f"base_pks={group['base_pks']}")
        print(f"pk_count before Use_Flag filter={len(pk_list)}")
        print("=" * 100)

        jac, df = estimate_jacobians_per_gray(
            pk_list=pk_list,
            set_mapping=mapping,
            lam=lam,
            delta_window=delta_window,
            gauss_sigma=gauss_sigma,
            min_samples=min_samples,
        )

        out_csv, out_npy = make_group_paths(
            panel_maker=panel_maker,
            frame_rate=frame_rate,
            lam=lam,
            delta_window=delta_window,
            gauss_sigma=gauss_sigma,
        )

        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        J_dense = np.full((256, 3, 3), np.nan, dtype=np.float32)
        n_arr = np.zeros(256, dtype=np.int32)
        condArr = np.full(256, np.nan, dtype=np.float32)

        for g, payload in jac.items():
            J_dense[g, :, :] = payload["J"]
            n_arr[g] = int(payload["n"])
            condArr[g] = float(payload["cond"])

        bundle = {
            "J": J_dense,
            "n": n_arr,
            "cond": condArr,
            "panel_maker": panel_maker,
            "frame_rate": frame_rate,
            "model_names": group["model_names"],
            "base_pks": group["base_pks"],
            "ref_pks": group["ref_pks"],
            "pk_list": pk_list,
            "lam": lam,
            "delta_window": delta_window,
            "gauss_sigma": gauss_sigma,
            "gray_used": [2, 253],
            "exclude_gray_for_cxcy": [0, 5],
            "schema": "J[gray, out(Cx,Cy,Gamma), in(R_High,G_High,B_High)]",
            "reference_mode": "base",
        }

        np.save(out_npy, bundle, allow_pickle=True)

        print(f"[OK] CSV saved -> {out_csv}")
        print(f"[OK] NPY saved -> {out_npy}")

        for g in (32, 128, 224):
            if np.isfinite(J_dense[g]).any():
                print(f"\n[g={g}] n={n_arr[g]}, cond={condArr[g]:.2e}")
                print(J_dense[g])
            else:
                print(f"\n[g={g}] no estimate")

    elapsed = time.time() - start_time
    print(f"\n[ALL DONE] elapsed = {elapsed:.2f} sec")
    
=

def debug_print_XY_at_grays(
    pk_list,
    set_mapping,
    grays=(0, 1, 64, 128, 192, 254, 255),
    max_rows_per_gray=5,
):
    X, Y0, groups, idx_gray, ds = build_white_X_Y0(
        pk_list=pk_list,
        set_mapping=set_mapping,
    )

    sample_ref_map = {
        int(s["pk"]): int(s["ref_pk"])
        for s in ds.samples
    }

    gray_norm = X[:, idx_gray]
    gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    print("\n================ DEBUG: X/Y rows at selected grays ================")
    print(f"pk_list_size={len(pk_list)}, collected_samples={len(ds.samples)}, total_rows={len(X)}")
    print(f"idx_gray={idx_gray}")
    print(f"target grays={list(grays)}")
    print("reference_mode=base")
    print("-------------------------------------------------------------------")

    for g in grays:
        m = gray_idx == int(g)
        n = int(m.sum())
        print(f"\n[gray={g}] rows={n}")

        if n == 0:
            continue

        idxs = np.where(m)[0][:max_rows_per_gray]

        for i in idxs:
            pk = int(groups[i])
            base_pk = sample_ref_map.get(pk)

            x_row = X[i]
            y_row = Y0[i]

            dR, dG, dB = float(x_row[0]), float(x_row[1]), float(x_row[2])
            dCx, dCy, dGam = float(y_row[0]), float(y_row[1]), float(y_row[2])

            print(f"  - row={i}, pk={pk}, base_pk={base_pk}")
            print(f"    X[0:3] (dR_H,dG_H,dB_H) = ({dR:+.3f}, {dG:+.3f}, {dB:+.3f})")
            print(f"    X(full) = {np.array2string(x_row, precision=4, floatmode='fixed')}")
            print(f"    Y (dCx,dCy,dGamma) = ({dCx:+.6f}, {dCy:+.6f}, {dGam:+.6f})")

    print("\n===================================================================\n")