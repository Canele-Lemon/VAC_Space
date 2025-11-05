class ReadVACdataThread(QThread):
    data_read = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, parent, vac_data_path=None, ser_tv=None, vacdataName=None):
        super().__init__(parent)
        self.parent = parent
        self.vac_data_path = vac_data_path
        self.ser_tv = ser_tv
        self.vacdataName = vacdataName

    def run(self):
        try:
            vac_debug_path = "/mnt/lg/cmn_data/panelcontroller/db/vac_debug"
            self.parent.send_command(self.ser_tv, 's')
            output = self.parent.check_directory_exists(vac_debug_path)
            
            if output == "exists":
                vac_data_path = vac_debug_path
            elif output == 'not_exists':
                vac_data_path = "/etc/panelcontroller/db/vac"
            else:
                self.error_occurred.emit(f"Error checking VAC debug path: {output}")
                return
            
            vacparam = self.parent.send_command(self.ser_tv, f'cat {vac_data_path}/{self.vacdataName}', output_limit=1000)
            
            if vacparam:
                vacparam = self._clean_vac_output(vacparam)
                vacparam = json.loads(vacparam)
                self.data_read.emit(vacparam)
            else:
                self.error_occurred.emit("VAC data read failed: empty response")
        except json.JSONDecodeError as e:
            self.error_occurred.emit(f"JSON decode error while reading VAC data: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error while reading VAC data: {e}")

    def _clean_vac_output(self, raw_output):
        cleaned = re.sub(r'^.*?\n\s*', '', raw_output)
        cleaned = re.sub(r'(?m)^\s*$\n', '', cleaned)
        cleaned = cleaned.replace("/ #", "").strip()
        return cleaned

지금 여기서는 exit 안하고 끝내는거죠?
