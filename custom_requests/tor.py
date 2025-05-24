import http.client
import socket
import time

import socks
from stem import Signal
from stem.control import Controller

CONTROLLER = None
CONNEXION = False
TIME_AFTER_RENEWING_TOR = 2

socks.set_default_proxy()


def renew_tor(time_after_newewing_tor=TIME_AFTER_RENEWING_TOR):
    """Create a connexion to Tor or renew it if it already exist"""
    global CONNEXION
    global CONTROLLER
    if not CONNEXION:
        CONTROLLER = Controller.from_port(port=9051)
        CONNEXION = True
    CONTROLLER.authenticate()
    CONTROLLER.signal(Signal.NEWNYM)
    socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", 9150, True)
    socket.socket = socks.socksocket
    time.sleep(time_after_newewing_tor)


def checkIP():
    conn = http.client.HTTPConnection("icanhazip.com")
    conn.request("GET", "/")
    time.sleep(3)
    response = conn.getresponse()
    return response.read()


if __name__ == "__main__":
    for i in range(3):
        renew_tor()
        print(f"{checkIP()=}")
