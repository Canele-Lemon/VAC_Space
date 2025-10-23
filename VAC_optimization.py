def start_VAC_optimization(self):
    base = cf.get_normalized_path(__file__, '..', '..', 'resources/images/pictures')
    self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
    self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
    self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))
    # ... 이하 기존 코드 유지 ....