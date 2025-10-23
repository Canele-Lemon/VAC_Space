        self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
        self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
        self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))

아이콘이 라벨에 로드될때 라벨 크기에 맞춰서 리사이즈되는게 아니라 본래 사이즈로 나오는거 같습니다. 기존 라벨사이즈에 맞게 리사이즈되도록 하려면 어떻게 하나요?
