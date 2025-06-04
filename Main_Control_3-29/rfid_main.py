import time
import RPi.GPIO as GPIO
import rfid_reader_v2 as rfid_reader
import servo_control_v2 as servo_control
import threading
from rfid_reader_v2 import exit_event

# Initialize GPIO mode
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # Using BCM mode to match your direct pin connections

# Global variables
rfid_success = False
card = None

def rfid_authentication():
    global rfid_success, card
    while not rfid_success:
        card_id = rfid_reader.read_rfid()
        print(f"Scanned card ID: {card_id}")

        if card_id is not None:
            rfid_success = True
            card = card_id
            if card_id == '860338929300':  # Your authorized card ID
                print('RFID verification successful')
                servo_control.unlock()
            else:
                print('Unauthorized card!', card_id)
            return

try:
    while True:
        print("Please scan your RFID card...")
        rfid_thread = threading.Thread(target=rfid_authentication)
        rfid_thread.start()
        rfid_thread.join()

        if rfid_success and card == '860338929300':
            print("The door will lock in 5 seconds!")
            time.sleep(5)
            servo_control.lock()
            print("The door has been locked!")
        
        # Reset for next scan
        rfid_success = False
        card = None
        time.sleep(1)

except KeyboardInterrupt:
    print('Program interrupted. Cleaning up GPIO settings...')

finally:
    exit_event.set()
    servo_control.cleanup()
    GPIO.cleanup()