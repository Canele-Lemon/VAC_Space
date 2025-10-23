from PySide2.QtCore import Qt

def _set_icon_scaled(self, label, pixmap):
    """라벨 현재 크기에 맞춰 아이콘 스케일 후 세팅"""
    if not pixmap or pixmap.isNull():
        label.clear(); return
    target_size = label.size()
    if target_size.width() <= 0 or target_size.height() <= 0:
        # 레이아웃 직후 1프레임 뒤로 미루고 스케일 (안전장치)
        from PySide2.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._set_icon_scaled(label, pixmap))
        return
    scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(scaled)