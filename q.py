def main():
    if use_gui:
        root = tk.Tk(); root.withdraw()
        
    # 1) 희소 High CSV 선택
    if use_gui:
        sparse_csv = askopenfilename(
            title="Gray-LUT_j-High(희소) CSV 선택",
            filetypes=[("CSV Files","*.csv"),("All Files","*.*")]
        )
        if not sparse_csv:
            print("@INFO: 입력 파일을 선택하지 않아 종료합니다.")
            return
    else:
        sparse_csv = input("희소 High CSV 경로: ").strip()

    # 2) 256 그레이 테이블 생성
    df_gray = build_gray_table(sparse_high_csv=sparse_csv, low_4096_csv=LOW_LUT_CSV)

    # 3) 중간 CSV 저장 (원하면 경로 선택)
    if use_gui:
        save_csv = asksaveasfilename(
            title="병합 LUT(256) CSV 저장",
            defaultextension=".csv",
            initialfile="LUT_gray_merged_256.csv",
            filetypes=[("CSV Files","*.csv")]
        )
        if not save_csv:
            # 자동 접미사 저장
            base, _ = os.path.splitext(sparse_csv)
            save_csv = f"{base}_merged_256.csv"
    else:
        base, _ = os.path.splitext(sparse_csv)
        save_csv = f"{base}_merged_256.csv"

    df_gray.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 256 그레이 LUT CSV 저장: {save_csv}")

    # 4) JSON 파일명 = "파일 대화상자로 고른 입력 CSV 이름".json
    #    (예: 입력이 D:\...\MyHighLUT.csv 라면 → D:\...\MyHighLUT.json)
    input_base = os.path.splitext(os.path.basename(sparse_csv))[0]
    out_json_path = os.path.join(os.path.dirname(save_csv), f"{input_base}.json")

    with open(out_json_path, 'w', encoding='utf-8') as f:
        # 👉 JSON 포맷으로 쓰도록 포맷 인자 명시
        write_default_data(f, table_format="json")
        write_LUT_data(f, input_file_path=save_csv, table_format="json")

    print(f"[OK] JSON 생성 완료: {out_json_path}")

    # Windows에서 열기 시도
    try:
        os.startfile(out_json_path)  # type: ignore[attr-defined]
    except Exception:
        pass

    if use_gui:
        messagebox.showinfo("완료", f"CSV/JSON 생성 완료\n\nCSV: {save_csv}\nJSON: {out_json_path}")