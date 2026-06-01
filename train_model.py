test_result = artifacts["test_result"]

df_detail, df_summary = report_metrics_by_set(
    mapping=dataset.set_mapping,
    groups_test=test_result["groups_test"],
    y_test=test_result["y_test"],
    y_pred=test_result["y_pred"],
    title=f"Y0-{comp}"
)

"set_metrics": df_summary.to_dict(orient="records"),

test_result = artifacts["test_result"]

df_detail, df_summary = report_metrics_by_set(
    mapping=dataset.set_mapping,
    groups_test=test_result["groups_test"],
    y_test=test_result["y_test"],
    y_pred=test_result["y_pred"],
    title="Y1-slope"
)

"set_metrics": df_summary.to_dict(orient="records"),

test_result = artifacts["test_result"]

df_detail, df_summary = report_metrics_by_set(
    mapping=dataset.set_mapping,
    groups_test=test_result["groups_test"],
    y_test=test_result["y_test"],
    y_pred=test_result["y_pred"],
    title="Y2-delta_uv"
)

"set_metrics": df_summary.to_dict(orient="records"),

