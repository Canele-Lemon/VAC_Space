def _panel_text_to_onehot(self, panel_text: str) -> np.ndarray:
    PANEL_MAKER_CATEGORIES = [['HKC(H2)', 'HKC(H5)', 'BOE', 'CSOT', 'INX']]
    cats = PANEL_MAKER_CATEGORIES[0]

    v = np.zeros(len(cats), dtype=np.float32)

    if panel_text is None:
        return v

    p = str(panel_text).strip()
    if p in cats:
        v[cats.index(p)] = 1.0

    return v