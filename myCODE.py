# 필요한 모듈 (파일 상단 어딘가에)
import os, json, numpy as np, tempfile, logging, platform, subprocess
from collections import OrderedDict
from PySide2.QtWidgets import QFileDialog, QMessageBox

def _dev_zero_lut_from_file(self):
    """원본 VAC JSON을 골라 6개 LUT 키만 0으로 덮어쓴 JSON을 임시파일로 저장하고 자동으로 엽니다."""
    # 1) 원본 JSON 선택
    fname, _ = QFileDialog.getOpenFileName(
        self, "원본 VAC JSON 선택", "", "JSON Files (*.json);;All Files (*)"
    )
    if not fname:
        return

    try:
        # 2) 순서 보존 로드
        with open(fname, "r", encoding="utf-8") as f:
            raw_txt = f.read()
        vac_dict = json.loads(raw_txt, object_pairs_hook=OrderedDict)

        # 3) LUT 6키를 모두 0으로 구성 (4096 포인트)
        zeros = np.zeros(4096, dtype=np.int32)
        new_lut = {
            "RchannelLow":  zeros,
            "RchannelHigh": zeros,
            "GchannelLow":  zeros,
            "GchannelHigh": zeros,
            "BchannelLow":  zeros,
            "BchannelHigh": zeros,
        }

        # 4) 최종 JSON 문자열 생성 (키 순서 보존 + 탭 들여쓰기)
        #    ※ build_vacparam_std_format 시그니처가 (base, new_lut)만 받는다면 pretty/use_tabs 인자를 빼세요.
        try:
            out_json = self.build_vacparam_std_format(
                base_vac_dict=vac_dict,
                new_lut_tvkeys=new_lut,
                pretty=True,
                use_tabs=True
            )
        except TypeError:
            # pretty/use_tabs 매개변수가 없다면 기본 호출로 처리
            out_json = self.build_vacparam_std_format(
                base_vac_dict=vac_dict,
                new_lut_tvkeys=new_lut
            )

        # (선택) 캐시도 동일 내용으로 갱신
        self._vac_dict_cache = json.loads(out_json, object_pairs_hook=OrderedDict)

        # 5) 임시파일로 저장
        fd, tmp_path = tempfile.mkstemp(prefix="VAC_zero_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(out_json)

        # 6) 플랫폼별 기본 앱으로 자동 열기
        try:
            sysname = platform.system()
            if sysname == "Windows":
                os.startfile(tmp_path)
            elif sysname == "Darwin":  # macOS
                subprocess.call(["open", tmp_path])
            else:  # Linux
                subprocess.call(["xdg-open", tmp_path])
        except Exception as e:
            logging.warning(f"임시파일 자동 열기 실패: {e}")

        QMessageBox.information(self, "완료", f"Zero-LUT JSON 임시파일 생성 및 열기 완료:\n{tmp_path}")

    except Exception as e:
        logging.exception(e)
        QMessageBox.critical(self, "오류", f"처리 중 오류: {e}")
        
        
from collections import OrderedDict

def build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict,
                              pretty: bool = False, use_tabs: bool = True) -> str:
    """
    base_vac_dict: TV에서 읽은 '원본 키' 그대로의 dict(제어필드 포함, 키 순서 중요)
    new_lut_tvkeys: 교체할 6채널만 TV 원본 키명으로 제공
      {"RchannelLow":[4096], "RchannelHigh":[4096],
       "GchannelLow":[4096], "GchannelHigh":[4096],
       "BchannelLow":[4096], "BchannelHigh":[4096]}
    """
    if not isinstance(base_vac_dict, dict):
        raise ValueError("base_vac_dict must be dict")

    # ✅ 순서 보존을 위해 OrderedDict로 얕은 복사
    out = OrderedDict(base_vac_dict)

    for k in ("RchannelLow","RchannelHigh","GchannelLow","GchannelHigh","BchannelLow","BchannelHigh"):
        if k in new_lut_tvkeys:
            arr = np.asarray(new_lut_tvkeys[k])
            if arr.shape != (4096,):
                raise ValueError(f"{k}: 길이는 4096이어야 합니다. (현재 {arr.shape})")
            out[k] = np.clip(np.round(arr).astype(np.int32), 0, 4095).tolist()

    if not pretty:
        return json.dumps(out, ensure_ascii=False, separators=(',', ':'))

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if use_tabs:
        txt = txt.replace("\n  ", "\n\t")
    return txt