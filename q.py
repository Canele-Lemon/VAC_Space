if __name__ == "__main__":

    debug_cases = [
        {"target_pk": 4300, "ref_pk": 4254},  # 50QNED85 INX
        {"target_pk": 4000, "ref_pk": 3943},  # 50QNED85 HKC
        {"target_pk": 3700, "ref_pk": 3631},  # 43UT80 CSOT
        {"target_pk": 3400, "ref_pk": 3320},  # 43NANO80 HKC
        {"target_pk": 3100, "ref_pk": 3007},  # 50UB85 INX
    ]

    for case in debug_cases:
        print("\n" + "=" * 100)
        print(
            f"target_pk={case['target_pk']} "
            f"ref_pk={case['ref_pk']}"
        )

        builder = VACInputBuilder(pk=case["target_pk"])

        builder.debug_dump_delta_with_mapping(
            ref_pk=case["ref_pk"],
            verbose_lut=True,
            preview_grays=[0, 1, 32, 128, 254, 255]
        )
        
        
        
# vac_set_mapping.py
import os
import pandas as pd


class VACSetMapping:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "data",
                    "vac_set_mapping.csv"
                )
            )

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        required_cols = [
            "pk_start", "pk_end", "ref_pk", "base_pk",
            "model_name", "panel_maker", "frame_rate", "model_year"
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"[VACSetMapping] Missing columns: {missing}")

    def get_row(self, pk: int):
        matched = self.df[
            (self.df["pk_start"].astype(int) <= pk) &
            (pk <= self.df["pk_end"].astype(int))
        ]

        if matched.empty:
            raise KeyError(f"[VACSetMapping] No mapping found for PK={pk}")

        if len(matched) > 1:
            raise ValueError(f"[VACSetMapping] Multiple mappings found for PK={pk}")

        return matched.iloc[0]

    def get_ref_pk(self, pk: int) -> int:
        return int(self.get_row(pk)["ref_pk"])

    def get_base_pk(self, pk: int) -> int:
        return int(self.get_row(pk)["base_pk"])

    def build_target_pk_list(self):
        pk_list = []
        for _, row in self.df.iterrows():
            pk_list.extend(range(int(row["pk_start"]), int(row["pk_end"]) + 1))
        return pk_list

    def summary(self):
        print(f"\n[VACSetMapping] csv_path = {self.csv_path}")
        print(f"[VACSetMapping] number of sets = {len(self.df)}")
        print(self.df.to_string(index=False))