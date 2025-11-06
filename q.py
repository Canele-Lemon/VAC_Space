--- R_Low ---  shape=(256,), dtype=float32
  gray   0 @ j=   0 : Δ= 0.000
  gray   1 @ j=   0 : Δ= 0.000
  gray  32 @ j= 499 : Δ= 0.000
  gray 128 @ j=2043 : Δ= 0.000
  gray 255 @ j=4092 : Δ= 0.000

--- R_High ---  shape=(256,), dtype=float32
  gray   0 @ j=   0 : Δ= 0.000
  gray   1 @ j=   0 : Δ= 0.000
  gray  32 @ j= 499 : Δ= 0.000
  gray 128 @ j=2043 : Δ= 0.000
  gray 255 @ j=4092 : Δ= 0.000

--- G_Low ---  shape=(256,), dtype=float32
  gray   0 @ j=   0 : Δ= 0.000
  gray   1 @ j=   0 : Δ= 0.000
  gray  32 @ j= 499 : Δ= 0.000
  gray 128 @ j=2043 : Δ= 0.000
  gray 255 @ j=4092 : Δ= 0.000

--- G_High ---  shape=(256,), dtype=float32
  gray   0 @ j=   0 : Δ= 0.000
  gray   1 @ j=   0 : Δ= 0.000
  gray  32 @ j= 499 : Δ= 25.000
  gray 128 @ j=2043 : Δ= 25.000
  gray 255 @ j=4092 : Δ= 0.000

--- B_Low ---  shape=(256,), dtype=float32
  gray   0 @ j=   0 : Δ= 0.000
  gray   1 @ j=   0 : Δ= 0.000
  gray  32 @ j= 499 : Δ= 0.000
  gray 128 @ j=2043 : Δ= 0.000
  gray 255 @ j=4092 : Δ= 0.000

--- B_High ---  shape=(256,), dtype=float32
  gray   0 @ j=   0 : Δ= 0.000
  gray   1 @ j=   0 : Δ= 0.000
  gray  32 @ j= 499 : Δ=-125.000
  gray 128 @ j=2043 : Δ=-125.000
  gray 255 @ j=4092 : Δ=-3.000

문제가 되는 부분은 위 디버그에서 B_High예요.
G에만 sweep offset을 +25 주었기 때문에 G_High만 변해야 하는데 B_High도 변합니다. 원인 파악을 위해 B_Hihg의 LUT값도 디버깅해볼 수 있을까요? 왜 저 값이 나왔는지 따져보려고요
