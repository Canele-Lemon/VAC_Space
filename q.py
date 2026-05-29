def debug_training_pk_mapping(dataset, n_head=5, n_tail=5):
    print("\n[DEBUG] Training PK mapping summary")

    rows = []
    for pk in dataset.pk_list:
        ref_pk = dataset.set_mapping.get_ref_pk(pk)
        base_pk = dataset.set_mapping.get_base_pk(pk)
        row = dataset.set_mapping.get_row(pk)

        rows.append({
            "pk": pk,
            "ref_pk": ref_pk,
            "base_pk": base_pk,
            "model_name": row["model_name"],
            "panel_maker": row["panel_maker"],
            "frame_rate": row["frame_rate"],
        })

    df = pd.DataFrame(rows)

    print("\n[DEBUG] PK count by set:")
    print(
        df.groupby(["model_name", "panel_maker", "frame_rate", "ref_pk", "base_pk"])
          .size()
          .reset_index(name="n_pk")
          .to_string(index=False)
    )

    print("\n[DEBUG] First/last PKs per set:")
    for _, sub in df.groupby(["model_name", "panel_maker", "frame_rate", "ref_pk", "base_pk"]):
        sub = sub.sort_values("pk")
        print("\n", sub[["pk", "ref_pk", "base_pk", "model_name", "panel_maker", "frame_rate"]]
              .head(n_head)
              .to_string(index=False))
        if len(sub) > n_head:
            print("...")
            print(sub[["pk", "ref_pk", "base_pk", "model_name", "panel_maker", "frame_rate"]]
                  .tail(n_tail)
                  .to_string(index=False))
                

학습 전 shape 확인
channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

for comp in ("dGamma", "dCx", "dCy"):
    X_dbg, y_dbg, g_dbg = dataset.build_XY_dataset(
        target="y0",
        component=comp,
        channels=channels,
        patterns=('W',),
    )
    preview(f"Y0-{comp}", X_dbg, y_dbg, g_dbg, n=3)

X_dbg, y_dbg, g_dbg = dataset.build_XY_dataset(
    target="y1",
    channels=channels,
    patterns=('W',),
)
preview("Y1", X_dbg, y_dbg, g_dbg, n=3)

X_dbg, y_dbg, g_dbg = dataset.build_XY_dataset(
    target="y2",
    channels=channels,
)
preview("Y2", X_dbg, y_dbg, g_dbg, n=3)