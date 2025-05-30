import http.client
import socket
import time
from typing import Optional, Any

import socks
from stem import Signal
from stem.control import Controller

CONTROLLER: Optional[Controller] = None
CONNEXION: bool = False
TIME_AFTER_RENEWING_TOR: int = 2

# The instruction mentions that socks.set_default_proxy() might be problematic here.
# However, the file structure is preserved.
socks.set_default_proxy()


def renew_tor(time_after_renewing_tor: int = TIME_AFTER_RENEWING_TOR) -> None:
    """Create a connexion to Tor or renew it if it already exist"""
    global CONNEXION
    global CONTROLLER
    try:
        if not CONNEXION or CONTROLLER is None:
            CONTROLLER = Controller.from_port(port=9051) # Default address: '127.0.0.1'
            CONTROLLER.authenticate() # Authenticate with no password, or add password if configured
            CONNEXION = True
        
        if CONTROLLER:
            CONTROLLER.signal(Signal.NEWNYM)
        
        # Configure default socket to use Tor SOCKS proxy
        # Port 9050 is standard for Tor SOCKS proxy, 9150 is often for Tor Browser's SOCKS proxy.
        # Using 9050 as it's more common for direct service interaction.
        socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", 9050, True)
        socket.socket = socks.socksocket # type: ignore [misc] # Monkeypatching socket; [misc] for mypy if strict
        
        time.sleep(time_after_renewing_tor)
    except Exception as e:
        # Log error or handle failure to connect/renew Tor
        # For now, just printing, but a logger would be better.
        print(f"Error renewing Tor: {e}")
        # Optionally, set CONNEXION to False or re-raise
        CONNEXION = False # Reset connection status on error


def checkIP() -> Optional[bytes]:
    """Checks the current external IP address."""
    try:
        # Using a known service that returns only the IP address.
        conn: http.client.HTTPConnection = http.client.HTTPConnection("icanhazip.com", timeout=10)
        conn.request("GET", "/")
        response: http.client.HTTPResponse = conn.getresponse()
        if response.status == 200:
            ip_address: bytes = response.read().strip()
            conn.close()
            return ip_address
        else:
            conn.close()
            return None
    except Exception as e:
        return None


if __name__ == "__main__":
    for i in range(3):
        print("Attempting to renew Tor IP...")
        renew_tor()
        print("Checking IP address...")
        current_ip_bytes: Optional[bytes] = checkIP()
        if current_ip_bytes is not None:
            print(f"Current IP: {current_ip_bytes.decode(errors='ignore')}")
        else:
            print("Could not retrieve current IP.")
        
        if i < 2: # Avoid sleeping after the last check
            print(f"Sleeping for {TIME_AFTER_RENEWING_TOR + 3} seconds before next check...")
            time.sleep(TIME_AFTER_RENEWING_TOR + 3) # Wait a bit longer to allow IP to change
