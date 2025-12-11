dataset = VACDataset(pk_list, ref_pk=bypass_pk)

# Y0 - dGamma용
X_dG, y_dG, grp_dG = dataset.build_XY_dataset(
    target="Y0",
    component="dGamma",
    channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
    patterns=('W',),   # 필요시 ('W','R','G','B')도 가능
)