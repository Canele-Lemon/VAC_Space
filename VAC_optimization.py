from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPixmap

# ===== 공통: 라벨 크기에 맞춰 스케일 후 세팅 =====
def _set_icon_scaled(self, label, pixmap: QPixmap):
    if not label or pixmap is None or pixmap.isNull():
        return
    size = label.size()
    if size.width() <= 0 or size.height() <= 0:
        # 라벨이 아직 레이아웃되기 전이면 다음 프레임에 재시도
        QTimer.singleShot(0, lambda: self._set_icon_scaled(label, pixmap))
        return
    scaled = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(scaled)

# ===== 내부용: step→라벨 위젯 찾기 =====
def _step_label(self, step: int):
    # UI에 있는 라벨 이름 규칙: vac_label_pixmap_step_{n}
    return getattr(self.ui, f"vac_label_pixmap_step_{step}", None)

# ===== 내부용: 진행중 애니메이션 핸들(라벨, 무비) 보관 =====
def _ensure_step_anim_map(self):
    if not hasattr(self, "_step_anim"):
        self._step_anim = {}  # {step: (label, movie)}

# ===== 공개 API: Step 시작/완료/실패 =====
def _step_start(self, step: int):
    """해당 단계의 '처리중 GIF' 시작"""
    self._ensure_step_anim_map()
    lbl = self._step_label(step)
    if lbl is None:
        return
    # 이미 돌아가는 중이면 무시
    if step in self._step_anim:
        return
    label_handle, movie_handle = self.start_loading_animation(lbl, 'processing.gif')
    self._step_anim[step] = (label_handle, movie_handle)

def _step_done(self, step: int):
    """해당 단계 애니 정지 + 완료 아이콘(스케일)"""
    self._ensure_step_anim_map()
    lbl = self._step_label(step)
    if lbl is None:
        return
    # 애니 정지
    if step in self._step_anim:
        try:
            label_handle, movie_handle = self._step_anim.pop(step)
            self.stop_loading_animation(label_handle, movie_handle)
        except Exception:
            pass
    # 완료 아이콘(라벨 크기에 맞춰)
    self._set_icon_scaled(lbl, self.process_complete_pixmap)

def _step_fail(self, step: int):
    """해당 단계 애니 정지 + 실패 아이콘(스케일)"""
    self._ensure_step_anim_map()
    lbl = self._step_label(step)
    if lbl is None:
        return
    # 애니 정지
    if step in self._step_anim:
        try:
            label_handle, movie_handle = self._step_anim.pop(step)
            self.stop_loading_animation(label_handle, movie_handle)
        except Exception:
            pass
    # 실패 아이콘(라벨 크기에 맞춰)
    self._set_icon_scaled(lbl, self.process_fail_pixmap)

def _step_set_pending(self, step: int):
    """대기(보류) 아이콘으로 교체"""
    lbl = self._step_label(step)
    if lbl is None:
        return
    self._set_icon_scaled(lbl, self.process_pending_pixmap)