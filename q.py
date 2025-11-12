    # Locks: Gray 0, 1  → 0 / Gray 254 → 4092 / Gray 255  → 4095
    mask_lowgray = (gray8 == 0 | gray8 == 1)
    mask_gray254 = (gray8 == 254)
    mask_gray255 = (gray8 == 255)

위 그레이 값들도 lock 해야 하거든요. 코드를 어떻게 써야 하나요? 마찬가지로 offset,eps,monotone 처리할때도 항상 고정입니다.
