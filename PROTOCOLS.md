# Protocols

## Hardware Serial Protocol
Data is sent through serial at a baud rate of 115200.
- GPS Protocol
    - Send a capital “G”
    - Receive: “A,B,C,D,E”
    - A = latitude
    - B = N/S
    - C = longitude
    - D = W/E
    - E = speed in knots
<br/><br/>
- Accelerometer Protocol
    - Send a capital “C” (int value 67)
    - Receive: “A,B,C”
    - A = compass reading, double value
    - B = system calibration, integer 0-3
    - C = magnetometer calibration, integer 0-3


