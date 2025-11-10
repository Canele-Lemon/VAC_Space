import os
import datetime

def _save_batch_corr_df(self, iter_idx: int, df_corr):
    """
    각 보정 회차별 df_corr CSV를 현재 파일(__file__)과 같은 폴더 내 artifacts 디렉터리에 저장.
    """
    # 1) 현재 파일 경로 기준으로 artifacts 폴더 지정
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    artifacts_dir = os.path.join(current_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)  # 없으면 생성

    # 2) 파일 이름 구성 (예: batch_corr_iter01_20251110_134500.csv)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_corr_iter{iter_idx:02d}_{ts}.csv"
    out_csv = os.path.join(artifacts_dir, filename)

    # 3) 저장
    df_corr.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 4) 로그 출력
    logging.info(f"[Batch Correction] CSV saved: {out_csv}")