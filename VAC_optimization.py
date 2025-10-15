    def start_VAC_optimization(self):
        
        
        # 1. VAC Control Off 하기 : VAC Control status 확인 후 On 이면 Off 후 측정 / Off 이면 바로 측정
        st = self.check_VAC_status()
        if st.get("activated", False):
            logging.debug("VAC 활성 상태 감지 → VAC OFF 시도")
            cmd = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff":false}\''
            self.send_command(self.ser_tv, 's')
            res = self.send_command(self.ser_tv, cmd)
            self.send_command(self.ser_tv, 'exit')
            logging.debug(f"VAC OFF 명령 응답: {res}")
            # 재확인
            st2 = self.check_VAC_status()
            if st2.get("activated", False):  # 여전히 True면 실패로 간주
                logging.warning("VAC OFF 실패로 보입니다. 그래도 측정은 진행합니다.")
            else:
                logging.info("VAC OFF 전환 성공.")
            return True
        else:
            logging.debug("이미 vac control off임. 측정 시작")
            return True
        
        # 2. VAC OFF일 때 측정
        
        # 3. VAC ON하고 DB에서 모델정보+패널주사율에 해당하는 VAC Data 가져와서 TV 적용 후 측정
        
        # 4. 보정 로직(자코비안 기반) 호출
        
        # 5. 재측정 (검증)
