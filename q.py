def _set_vac_active(self, enable: bool) -> bool:
    try:
        self.send_command(self.ser_tv, 's')
        cmd = (
            f"luna-send -n 1 -f "
            f"luna://com.webos.service.panelcontroller/setVACActive '{{\"OnOff\":{str(enable).lower()}}}'"
        )
        self.send_command(self.ser_tv, cmd)
        self.send_command(self.ser_tv, 'exit')
        time.sleep(0.5)
        st = self.check_VAC_status()
        return bool(st.get("activated", False)) == enable
    except Exception as e:
        logging.error(f"VAC {'ON' if enable else 'OFF'} 전환 실패: {e}")
        return False