import serial
import time


class GpsDriver:
    # /dev/ttyS4, 115200 *very special indeed*
    def __init__(self, prt="/dev/ttyS4", baud=2000000):
        self.serialGPS = serial.Serial(
            port=prt,
            baudrate=baud,
            timeout=1
        )

    def receiveData(self):
        self.serialGPS.write(str.encode("G"))
        return self.serialGPS.readline()

    def getCoordinates(self):  # lat, lon
        raw = self.receiveData()

        raw = raw[0:-2]
        tokens = raw.split(b',')
        # print(raw)
        try:
            return float(tokens[0]), float(tokens[2]), float(tokens[4])  # no data yet
        except:
            return 0.0, 0.0, 0.0