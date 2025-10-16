class GammaChart:
    ...
    # ▼ 추가
    def add_series(self, axis_index=0, label=None, color=None, linestyle='-'):
        key = f"{label or 'series'}_{axis_index}_{len(self.chart.lines)}"
        self.chart.add_line(key, color=color or 'black', linestyle=linestyle, label=label, axis_index=axis_index)
        return self.chart.lines[key]

    def autoscale(self):
        # XYChart가 relim/autoscale_view를 update에서 하긴 하지만, 외부에서 강제 호출용
        for ax in self.chart.axes:
            ax.relim(); ax.autoscale_view()
        self.canvas.draw_idle() if hasattr(self, 'canvas') else self.chart.canvas.draw_idle()

    def draw(self):
        self.chart.canvas.draw_idle()
        
        
class CIE1976ChromaticityDiagram:
    ...
    def _ensure_line(self, key):
        if key in self.lines:
            return
        # 데이터셋 별 색/마커 대충 구분 (원하시면 팔레트 바꾸세요)
        marker = 'o' if key.endswith('0deg') or '_0deg' in key else 's'
        # data_n에 따라 색 배정
        if 'data_1' in key: edge = 'red'
        elif 'data_2' in key: edge = 'green'
        else: edge = 'blue'
        line, = self.ax.plot([], [], marker, markerfacecolor='none', markeredgecolor=edge)
        self.lines[key] = line
        self.data[key] = {'u': [], 'v': []}

    def update(self, u_p, v_p, data_label, view_angle, vac_status):
        key = f'{data_label}_{view_angle}deg'
        # ▼ 추가
        if key not in self.lines:
            self._ensure_line(key)

        self.data[key]['u'].append(float(u_p))
        self.data[key]['v'].append(float(v_p))
        self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
        self.lines[key].set_label(f'Data #{data_label[-1]} {view_angle}° {vac_status}')
        self.ax.legend(fontsize=9)
        self.canvas.draw()
        
        
[Start Button]
     |
     v
start_VAC_optimization
  |-- load Jacobian (pkl) -> cache (self._jac_artifacts)
  |-- ensure VAC OFF (luna-send OnOff:false)
  v
_run_off_baseline_then_on
  |-- SessionProfile(OFF, table cols 0..3, ref=None)
  |-- gamma lines(OFF) 생성
  v
start_viewing_angle_session(OFF)
  |-- phase: 'gamma' -> (white,red,green,blue) x gray(0..255)
  |     |-- changeColor(rgb)
  |     |-- MeasureThread(main[/sub]) -> _consume_gamma_pair -> 차트/테이블 기록
  |
  |-- phase: 'colorshift' -> 맥베스 패턴 순회
  |     |-- changeColor -> MeasureThread -> CIE1976 업데이트
  |
  '--> _finalize_session(OFF)
         |-- white(main) Lv -> Gamma 계산 -> 테이블 OFF 감마 열 기록
         '--> on_done(store_off) == _apply_vac_from_db_and_measure_on

_apply_vac_from_db_and_measure_on
  |-- DB에서 VAC_Data 가져오기
  |-- _write_vac_to_tv (동기)
  |-- _read_vac_from_tv (동기) -> LUT 차트/테이블 갱신
  |-- SessionProfile(ON, table cols 4..10, ref=OFF)
  v
start_viewing_angle_session(ON)  (동일 루프)
  '--> _finalize_session(ON)
         |-- white(main) Lv -> Gamma -> 테이블(ON 감마 열)
         |-- OFF 대비 ΔGamma/ΔCx/ΔCy 기록
         |-- _check_spec_pass
                |-- PASS -> 종료
                '-- FAIL -> _run_correction_iteration(1)

_run_correction_iteration(i)
  |-- _read_vac_from_tv -> 4096->256 다운샘플
  |-- OFF vs ON Δ 시작치(Target) 구성
  |-- A_Gamma/A_Cx/A_Cy 구성 -> 연립방정식 -> Δh
  |-- 256 보정곡선 High 채널에 적용 -> 12bit 업샘플 -> _write_vac_to_tv
  |-- _read_vac_from_tv -> LUT 차트 갱신
  |-- SessionProfile(CORR{i}, table cols 4..10, ref=OFF)
  v
start_viewing_angle_session(CORR{i})  (동일 루프)
  '--> _finalize_session(CORR{i})
        |-- Δ 계산, 스펙 확인
        |-- PASS or i == max_iters ? 종료 : _run_correction_iteration(i+1)
        
        
.