2025-11-05 19:34:22,205 - INFO - subpage_vacspace.py:1366 - [Correction] LUT 1차 보정 완료
2025-11-05 19:34:22,232 - DEBUG - subpage_vacspace.py:7498 - Sending command: s
2025-11-05 19:34:22,446 - DEBUG - subpage_vacspace.py:7529 - Output: s
/bin/sh: s: not found
/ #
/ #
2025-11-05 19:34:22,447 - DEBUG - subpage_vacspace.py:7498 - Sending command: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || echo 'not_exists'
2025-11-05 19:34:22,660 - DEBUG - subpage_vacspace.py:7529 - Output: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || ech
o 'not_exists'
exists
/ #
/ #
2025-11-05 19:34:22,660 - DEBUG - subpage_vacspace.py:7498 - Sending command: cp /etc/panelcontroller/db/vac/vac_INX_50_60hz.json /mnt/lg/cmn_data/panelcontroller/db/vac_debug
2025-11-05 19:34:22,873 - DEBUG - subpage_vacspace.py:7529 - Output: cp /etc/panelcontroller/db/vac/vac_INX_50_60hz.json /mnt/lg/cmn_data/panelco
ntroller/db/vac_debug

/ #
/ #
2025-11-05 19:34:35,863 - DEBUG - subpage_vacspace.py:7527 - Output (truncated): cat > /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
{
"DRV_valc_major_ctrl"   :       [       0,      1       ],
"DRV_valc_pattern_ctrl_0"       :       [       5,      1       ],
"DRV_valc_pattern_ctrl_1"       :       [       [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ]               ],
"DRV_valc_sat_ctrl"     :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0       ],
"DRV_valc_hpf_ctrl_0"   :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0       ],
"DRV_valc_hpf_ctrl_1"   :               1,
"RchannelLow"   :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                        0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                        0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                        0,      0,      0,      0,      0,      0
2025-11-05 19:34:35,864 - DEBUG - subpage_vacspace.py:7498 - Sending command: restart panelcontroller
2025-11-05 19:34:36,075 - DEBUG - subpage_vacspace.py:7529 - Output: restart panelcontroller
restart panelcontroller
/ #
/ #
2025-11-05 19:34:36,076 - DEBUG - subpage_vacspace.py:7498 - Sending command: exit
2025-11-05 19:34:36,289 - DEBUG - subpage_vacspace.py:7529 - Output: exit
exit shell mode

030818.948711:teminalmanager-h:help] 0x0000000a
030818.948733:teminalmanager-h:help] 0x0000000a
030818.948759:teminalmanager-h:help] 0x0000000a
2025-11-05 19:34:36,295 - INFO - subpage_vacspace.py:1376 - [VAC Writing] write result: True VAC data written to /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
2025-11-05 19:34:36,295 - INFO - subpage_vacspace.py:1379 - [VAC Reading] TV reading after write
2025-11-05 19:34:36,297 - DEBUG - subpage_vacspace.py:7498 - Sending command: s
2025-11-05 19:34:36,519 - DEBUG - subpage_vacspace.py:7529 - Output: 
030819.169486:teminalmanager-h:help] 0x00000073
start shell mode.
/ #
2025-11-05 19:34:36,520 - DEBUG - subpage_vacspace.py:7498 - Sending command: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || echo 'not_exists'
2025-11-05 19:34:36,732 - DEBUG - subpage_vacspace.py:7529 - Output: [ -d /mnt/lg/cmn_data/panelcontroller/db/vac_debug ] && echo 'exists' || ech
o 'not_exists'
exists
/ #
/ #
2025-11-05 19:34:36,733 - DEBUG - subpage_vacspace.py:7498 - Sending command: cat /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
2025-11-05 19:34:49,402 - DEBUG - subpage_vacspace.py:7527 - Output (truncated): cat /mnt/lg/cmn_data/panelcontroller/db/vac_debug/vac_INX_50_60hz.json
{
"DRV_valc_major_ctrl"   :       [       0,      1       ],
"DRV_valc_pattern_ctrl_0"       :       [       5,      1       ],
"DRV_valc_pattern_ctrl_1"       :       [       [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ],
                        [       1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0       ],
                        [       0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1       ]               ],
"DRV_valc_sat_ctrl"     :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0       ],
"DRV_valc_hpf_ctrl_0"   :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0       ],
"DRV_valc_hpf_ctrl_1"   :               1,
"RchannelLow"   :       [       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                        0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                        0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                        0,      0,      0,      0,      0,      0,
2025-11-05 19:34:49,413 - INFO - subpage_vacspace.py:1386 - [VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.
2025-11-05 19:34:49,414 - INFO - subpage_vacspace.py:2572 - [VAC Reading] VAC data successfully verified - no mismatches.
2025-11-05 19:34:49,415 - INFO - subpage_vacspace.py:1392 - [VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치
2025-11-05 19:34:49,426 - INFO - subpage_vacspace.py:1423 - [BATCH CORR] re-measure start (after LUT update)
2025-11-05 19:42:26,413 - INFO - subpage_vacspace.py:2779 - [FineNorm] updated from last ON: Lv0=0.040, denom=303.460
2025-11-05 19:42:26,426 - INFO - subpage_vacspace.py:778 - [Evaluation] max|ΔGamma|=0.467177 (≤0.05), max|ΔCx|=0.006600, max|ΔCy|=0.012700 (≤0.003), NG grays=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 95, 96, 97, 98, 99, 100, 101, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253]
2025-11-05 19:42:26,428 - DEBUG - subpage_vacspace.py:892 - 1차 보정 결과: Cx:251/256, Cy:221/256, Gamma:137/253
2025-11-05 19:42:26,639 - INFO - subpage_vacspace.py:832 - [Correction] 최대 보정 횟수 도달 — 종료
