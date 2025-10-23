def _step_start(self, idx: int):
    """idx=1..5: 단계 시작(GIF on)"""
    label_widget = getattr(self.ui, f"vac_label_pixmap_step_{idx}")
    label, movie = self.start_loading_animation(label_widget, 'processing.gif')
    setattr(self, f"label_processing_step_{idx}", label)
    setattr(self, f"movie_processing_step_{idx}", movie)

def _step_done(self, idx: int):
    """idx=1..5: 단계 종료(GIF off + 완료아이콘)"""
    label = getattr(self, f"label_processing_step_{idx}", None)
    movie = getattr(self, f"movie_processing_step_{idx}", None)
    if label is not None and movie is not None:
        self.stop_loading_animation(label, movie)
    # 완료 아이콘
    getattr(self.ui, f"vac_label_pixmap_step_{idx}").setPixmap(self.process_complete_pixmap)

.