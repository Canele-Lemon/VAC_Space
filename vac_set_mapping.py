# vac_set_mapping.py
import os
import pandas as pd

class VACSetMapping:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def get_row(self, pk: int):
        matched = self.df[
            (self.df["pk_start"] <= pk) &
            (pk <= self.df["pk_end"])
        ]

        if matched.empty:
            raise KeyError(f"No mapping found for PK={pk}")

        if len(matched) > 1:
            raise ValueError(f"Multiple mappings found for PK={pk}")

        return matched.iloc[0]

    def get_ref_pk(self, pk: int) -> int:
        return int(self.get_row(pk)["ref_pk"])

    def get_base_pk(self, pk: int) -> int:
        return int(self.get_row(pk)["base_pk"])

    def get_meta(self, pk: int) -> dict:
        row = self.get_row(pk)
        return {
            "model_name": row["model_name"],
            "panel_maker": row["panel_maker"],
            "frame_rate": float(row["frame_rate"]),
            "model_year": str(row["model_year"]),
        }

    def build_target_pk_list(self):
        pk_list = []
        for _, row in self.df.iterrows():
            pk_list.extend(range(int(row["pk_start"]), int(row["pk_end"]) + 1))
        return pk_list
        