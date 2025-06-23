HOST = ""
PORT = 12345
PICO_IPS = {
    "hotspot" : "192.168.4.1", 
    "base" : "192.168.4.56",
    "1" : "192.168.4.60",
    "2" : "192.168.4.61",
    }
# Position tracking tag IDs
ROOMBA_1_TAG_ID = "5620"  # This roomba
ROOMBA_2_TAG_ID = "4F2A"  # Opponent roomba (for now both are the same)
PS4_KEYS = {
    "x" : 0,
    "circle" : 1,
    "square" : 2,
    "triangle" : 3,
    "share" : 4,
    "ps" : 5,
    "options" : 6,
    "left_stick_click" : 7,
    "right_stick_click" : 8,
    "L1" : 9,
    "R1" : 10,
    "up_arrow" : 11,
    "down_arrow" : 12,
    "left_arrow" : 13,
    "right_arrow" : 14,
    "touchpad" : 15
    }
DEADBAND = 10
SERIAL_PORT = "/dev/tty.usbmodem0007601851681"
BAUD = 115200