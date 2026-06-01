def report_metrics_by_set(mapping, groups_test, y_test, y_pred, title=""):
    rows = []

    for pk, yt, yp in zip(groups_test, y_test, y_pred):
        row = mapping.get_row(int(pk))

        err = float(yp - yt)

        rows.append({
            "pk": int(pk),
            "model_name": row["model_name"],
            "panel_maker": row["panel_maker"],
            "frame_rate": row["frame_rate"],
            "ref_pk": int(row["ref_pk"]),
            "y_true": float(yt),
            "y_pred": float(yp),
            "err": err,
            "abs_err": abs(err),
            "sq_err": err ** 2,
        })

    df = pd.DataFrame(rows)

    group_cols = ["model_name", "panel_maker", "frame_rate", "ref_pk"]

    summary_rows = []
    for keys, sub in df.groupby(group_cols):
        y_true = sub["y_true"].to_numpy()
        y_hat = sub["y_pred"].to_numpy()
        err = sub["err"].to_numpy()

        if len(sub) >= 2 and np.nanstd(y_true) > 1e-12:
            r2 = r2_score(y_true, y_hat)
        else:
            r2 = np.nan

        summary_rows.append({
            "model_name": keys[0],
            "panel_maker": keys[1],
            "frame_rate": keys[2],
            "ref_pk": keys[3],
            "n": len(sub),
            "mae": np.mean(np.abs(err)),
            "rmse": np.sqrt(np.mean(err ** 2)),
            "r2": r2,
            "y_mean": np.mean(y_true),
            "y_std": np.std(y_true),
        })

    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 100)
    print(f"[SET METRIC] {title}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    return df, summary
    
final_mse = mean_squared_error(y_test, y_pred_hybrid)
final_rmse = np.sqrt(final_mse)
final_mae = mean_absolute_error(y_test, y_pred_hybrid)
final_r2  = r2_score(y_test, y_pred_hybrid)

print(
    f"🏁 [{tag}] Hybrid — "
    f"MAE:{final_mae:.6f} RMSE:{final_rmse:.6f} R²:{final_r2:.6f}"
)

"hybrid": {
    "mae": float(final_mae),
    "mse": float(final_mse),
    "rmse": float(final_rmse),
    "r2": float(final_r2)
}