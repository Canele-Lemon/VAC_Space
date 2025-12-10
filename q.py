예측모델: VAC input에 따른 Cx, Cy, Gamma 값과 VAC OFF일 때의 Cx, Cy, Gamma 값과의 차이인 dCx, dCy, dGamma을 학습함
자코비안: Base VAC 기준 +50~-50 sweep을 주었을 때 Cx, Cy, Gamma 값들의 변화. 각 gray level에서 VAC 변화에 따른 Cx, Cy, Gamma 변화량

-VAC 최적화 로직 flow-
1. VAC OFF 측정
2. DB에서 Base VAC 불러온 후 예측모델을 이용해 VAC OFF 대비 dCx, dCy, dGamma 예측 후 이를 스펙 in으로 만들기 위해 자코비안 보정하여 예측 VAC를 generate함. (|dCx/dCy|<=0.003, |Gamma|<=0.05)
3. 예측 VAC를 TV에 적용 후 측정.
4. 각 NG gray에서 스펙 in을 만들기 위해 자코비안을 이용해 미세 보정을 하고 보정한 VAC를 TV 적용 후 해당 Gray만 측정
5. NG gray가 없어질때까지 보정-TV적용-측정 반복
