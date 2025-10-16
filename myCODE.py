from PySide2.QtCore import QThread, Signal

class MeasureThread(QThread):
    measure_completed = Signal(str, tuple)  # role, result

    def __init__(self, inst_cls, role):
        super().__init__()
        self.inst_cls = inst_cls
        self.role = role
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if self._is_cancelled:
            self.measure_completed.emit(self.role, None)
            return

        try:
            result = self.inst_cls.measure()  # (x, y, lv, cct, duv)
            
            if self._is_cancelled:
                self.measure_completed.emit(self.role, None)
            else:
                self.measure_completed.emit(self.role, result)
                
        except Exception as e:
            self.measure_completed.emit(self.role, None)

      import json
import re
import time
from PySide2.QtCore import QThread, Signal

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

class WriteVACdataThread(QThread):
    write_finished = Signal(bool, str)

    def __init__(self, parent, ser_tv, vacdataName, vacdata_loaded):
        super().__init__(parent)
        self.parent = parent
        self.ser_tv = ser_tv
        self.vacdataName = vacdataName
        self.vacdata_loaded = vacdata_loaded

    def run(self):
        try:
            vac_debug_path = "/mnt/lg/cmn_data/panelcontroller/db/vac_debug"
            self.parent.send_command(self.ser_tv, 's')
            output = self.parent.check_directory_exists(vac_debug_path)

            if output == 'exists':
                pass
            elif output == 'not_exists':
                self.parent.send_command(self.ser_tv, f"mkdir -p {vac_debug_path}")
            else:
                self.parent.send_command(self.ser_tv, 'exit')
                self.write_finished.emit(False, f"Error checking VAC debug path: {output}")
                return

            copyVACdata = f"cp /etc/panelcontroller/db/vac/{self.vacdataName} {vac_debug_path}"
            self.parent.send_command(self.ser_tv, copyVACdata)

            if self.vacdata_loaded is None:
                self.parent.send_command(self.ser_tv, 'exit')
                self.write_finished.emit(False, "No VAC data loaded.")
                return
            
            writeVACdata = f'cat > {vac_debug_path}/{self.vacdataName}'
            self.ser_tv.write((writeVACdata + '\n').encode())
            time.sleep(0.1)
            self.ser_tv.write(self.vacdata_loaded.encode())
            time.sleep(0.1)
            self.ser_tv.write(b'\x04')  # Ctrl+D
            time.sleep(0.1)
            self.ser_tv.write(b'\x04')  # Ctrl+D
            time.sleep(0.1)
            self.ser_tv.flush()

            self.parent.read_output(self.ser_tv, output_limit=1000)

            self.parent.send_command(self.ser_tv, 'restart panelcontroller')
            self.parent.send_command(self.ser_tv, 'exit')

            self.write_finished.emit(True, f"VAC data written to {vac_debug_path}/{self.vacdataName}")
        except Exception as e:
            self.write_finished.emit(False, f"Unexpected error while writing VAC data: {e}")

