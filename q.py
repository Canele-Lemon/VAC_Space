    @staticmethod
    def debug_representative_sets(
        verbose_lut: bool = True,
        preview_grays: list[int] | None = None,
    ):
        """
        vac_set_mapping.csv 기준으로 각 set 대표 PK 1개씩 선택해서
        target_pk → ref_pk → ΔLUT feature가 정상 생성되는지 확인.
        """

        mapping = VACSetMapping()
        rep_rows = mapping.get_representative_pks()

        if preview_grays is None:
            preview_grays = [0, 1, 32, 128, 254, 255]

        print("\n" + "=" * 100)
        print("[DEBUG] Representative set input feature check")
        print(f"[DEBUG] mapping file: {mapping.csv_path}")
        print(f"[DEBUG] number of sets: {len(rep_rows)}")
        print("=" * 100)

        for row in rep_rows:
            target_pk = int(row["target_pk"])
            ref_pk = int(row["ref_pk"])
            base_pk = int(row["base_pk"])

            print("\n" + "#" * 100)
            print(
                f"[SET] model={row['model_name']} | "
                f"maker={row['panel_maker']} | "
                f"frame={row['frame_rate']} | "
                f"year={row['model_year']}"
            )
            print(f"[PK] target_pk={target_pk}, ref_pk={ref_pk}, base_pk={base_pk}")
            print("#" * 100)

            builder = VACInputBuilder(pk=target_pk)

            builder.debug_dump_delta_with_mapping(
                pk=target_pk,
                ref_pk=ref_pk,
                verbose_lut=verbose_lut,
                preview_grays=preview_grays,
            )