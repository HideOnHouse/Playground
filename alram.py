import time
import webbrowser
from datetime import datetime

"""
maple story blooming event alarm via web
"""


def main():
    while True:
        now = datetime.now()
        if now.minute == 14 or now.minute == 44:
            webbrowser.open("https://www.virustotal.com/gui/")
        else:
            time.sleep(10)


if __name__ == '__main__':
    main()
