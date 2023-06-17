import serial
import time


class RobotDriver:
    # /dev/ttyS4, 115200 *very special indeed*
    def __init__(self, prt="/dev/cu.usbserial-130", baud=115200):
        self.robotSerial = serial.Serial(
            port=prt,
            baudrate=baud,
            timeout=1
        )

    def setSteering(self, steering):
        self.robotSerial.write((str(steering)).encode())
        self.robotSerial.write(b"\r\n")

        print(str(steering).encode())