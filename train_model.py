artifacts = {
    "linear_model": linear_model,
    "rf_residual": best_rf,
    "rf_best_params": search.best_params_,
    "metrics": {
        "linear_only": {"mse": float(base_mse), "r2": float(base_r2)},
        "hybrid": {"mse": float(final_mse), "r2": float(final_r2)}
    },
    "target_scaler": {"mean": y_mean, "std": y_std, "standardized": bool(normalize_y)},

    "test_result": {
        "test_idx": test_idx.astype(int),
        "groups_test": groups[test_idx].astype(int),
        "y_test": y_test.astype(float),
        "y_pred": y_pred_hybrid.astype(float),
    }
}

from sklearn.metrics import mean_absolute_error

def report_metrics_by_set(mapping, groups_test, y_test, y_pred, title=""):
    rows = []

    for pk, yt, yp in zip(groups_test, y_test, y_pred):
        row = mapping.get_row(int(pk))

        rows.append({
            "pk": int(pk),
            "model_name": row["model_name"],
            "panel_maker": row["panel_maker"],
            "frame_rate": row["frame_rate"],
            "ref_pk": int(row["ref_pk"]),
            "y_true": float(yt),
            "y_pred": float(yp),
            "err": float(yp - yt),
            "abs_err": float(abs(yp - yt)),
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["model_name", "panel_maker", "frame_rate", "ref_pk"])
          .agg(
              n=("y_true", "count"),
              mae=("abs_err", "mean"),
              rmse=("err", lambda x: np.sqrt(np.mean(np.square(x)))),
              y_std=("y_true", "std"),
          )
          .reset_index()
    )

    print("\n" + "=" * 100)
    print(f"[SET METRIC] {title}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    return df, summary
    
    
test_result = artifacts["test_result"]

df_detail, df_summary = report_metrics_by_set(
    mapping=dataset.set_mapping,
    groups_test=test_result["groups_test"],
    y_test=test_result["y_test"],
    y_pred=test_result["y_pred"],
    title=f"Y0-{comp}"
)