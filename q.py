import tempfile
import os

def main():

    # 테스트: offset = +500 한 세트만 생성
    R_off = 500
    G_off = 500
    B_off = 500

    print(f"[TEST] Generating single LUT: R={R_off}, G={G_off}, B={B_off}")

    df = _build_single_lut(R_off, G_off, B_off)   # ← DataFrame 생성

    # 임시파일 생성
    tmp = tempfile.NamedTemporaryFile(delete=False,
                                      suffix=f"_LUT_R{R_off}_G{G_off}_B{B_off}.csv")
    tmp_path = tmp.name
    tmp.close()

    # CSV로 저장
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 임시 CSV 생성: {tmp_path}")

    # Windows에서 자동 열기
    try:
        os.startfile(tmp_path)
    except Exception:
        pass

    print("[DONE] Single LUT test completed.")