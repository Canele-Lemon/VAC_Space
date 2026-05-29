if __name__ == "__main__":

    pk_list = [4300, 4000, 3700, 3400, 3100]

    dataset = VACDataset(pk_list=pk_list)

    print("\n[DEBUG] collected samples")
    for s in dataset.samples:
        print(f"pk={s['pk']}, ref_pk={s['ref_pk']}")

    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    # -----------------------------
    # Y0 dataset preview
    # -----------------------------
    target = "y0"
    X, y, grp = dataset.build_XY_dataset(
        target=target,
        component="dCx",
        channels=channels,
        patterns=('W',),
    )

    preview_xy_dataset(
        dataset, X, y, grp,
        channels=channels,
        target=target,
        n=30
    )

    # -----------------------------
    # Y1 dataset preview
    # -----------------------------
    target = "y1"
    X, y, grp = dataset.build_XY_dataset(
        target=target,
        channels=channels,
        patterns=('W',),
    )

    preview_xy_dataset(
        dataset, X, y, grp,
        channels=channels,
        target=target,
        n=30
    )

    # -----------------------------
    # Y2 dataset preview
    # -----------------------------
    target = "y2"
    X, y, grp = dataset.build_XY_dataset(
        target=target,
        channels=channels,
    )

    preview_xy_dataset(
        dataset, X, y, grp,
        channels=channels,
        target=target,
        n=30
    )