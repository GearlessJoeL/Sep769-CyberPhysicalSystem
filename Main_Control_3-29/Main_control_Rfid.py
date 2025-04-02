import time
import RPi.GPIO as GPIO
import rfid_reader_v2 as rfid_reader
import led_control_v2 as led_control
import buzzer_control_v3 as buzzer_control
import servo_control_v2 as servo_control
import threading
from rfid_reader_v2 import exit_event
from face_recog import FaceRecognition
from pubnub.callbacks import SubscribeCallback
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
import json

# PubNub Configuration
pnconfig = PNConfiguration()
pnconfig.subscribe_key = 'sub-c-a6797b99-e665-4db1-b0ec-2cb77ad995ed'
pnconfig.publish_key = 'pub-c-e478cfb1-92ef-4faa-93cc-d1c4022ecb19'
pnconfig.uuid = '321'
pubnub = PubNub(pnconfig)

# Channel names
CONTROL_CHANNEL = "MingyiHUO728"
STATUS_CHANNEL = "MingyiHUO728"

# Initialize GPIO mode
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# Initialize modules
face = FaceRecognition()

rfid_success = False
face_success = False
face_fail = False
remote_unlock = False

class MySubscribeCallback(SubscribeCallback):
    def message(self, pubnub, message):
        global remote_unlock
        if message.channel == CONTROL_CHANNEL:
            if message.message.get('message_type') == "control" and message.message.get('action') == 'unlock':
                remote_unlock = True
                print("Received remote unlock command")
                led_control.led_success()
                buzzer_control.buzzer_success()
                servo_control.unlock()
                
                # Publish status update
                status_data = {
                    "message_type": "status",
                    "state": 1,
                    "type": "remote",
                    "time": time.time(),
                    "name": "Remote Access"
                }
                publish_status(status_data)
                
                # Auto-lock after 5 seconds
                time.sleep(5)
                servo_control.lock()
                
                # Update locked status
                status_data["state"] = 0
                status_data["time"] = time.time()
                publish_status(status_data)

pubnub.add_listener(MySubscribeCallback())
pubnub.subscribe().channels([CONTROL_CHANNEL]).execute()

def publish_status(status_data):
    status_data["message_type"] = "status"  # Add message type
    pubnub.publish().channel(STATUS_CHANNEL).message(status_data).sync()

def rfid_authentication():
    global rfid_success
    while not rfid_success and not face_success and not remote_unlock:
        if face_success or remote_unlock:
            print("Face recognition succeeded or remote unlock activated, RFID thread exiting.")
            exit_event.set()
            return

        led_control.led_waiting()
        card_id = rfid_reader.read_rfid()

        if card_id == '860338929300':
            print('RFID verification successful')
            led_control.led_success()
            buzzer_control.buzzer_success()
            servo_control.unlock()
            rfid_success = True
            return

def face_authentication():
    global face_success
    global face_fail
    print('please face the camera...')
    while not rfid_success and not face_success and not remote_unlock:
        face.recognize()
        time.sleep(0.5)
        if face.get_name() != "":
            if face.get_name() == "Unknown":
                fail_count += 1
#                 print(fail_count)
                if fail_count > 4:
                    status_data = {
                        "state": 0,
                        "type": "face",
                        "time": time.time(),
                        "name": "Unknown"
                    }
                    print("Unknown person detected")
                    publish_status(status_data)
                    fail_count = 0
            else:
                led_control.led_success()
                buzzer_control.buzzer_success()
                servo_control.unlock()
                face_success = True
                exit_event.set()
#                 print("ex")
                return

try:
    rfid_thread = threading.Thread(target=rfid_authentication)
    face_thread = threading.Thread(target=face_authentication)

    rfid_thread.start()
    face_thread.start()

    rfid_thread.join()
    face_thread.join()
    
    if face_success or rfid_success or remote_unlock: #or fingerprint_success
        status_data = {
            "state": 1,
            "type": "",
            "time": time.time(),
            "name": ""
        }

        if face_success:
            status_data["type"] = "face"
            status_data["name"] = face.get_name
        elif rfid_success:
            status_data["type"] = "rfid"
            status_data["name"] = "Key"
        elif remote_unlock:
            status_data["type"] = "remote"
            status_data["name"] = "Remote Access"
        # elif fingerprint_success:
        #     status_data["type"] = "fingerprint"
        #     status_data["name"] = "Fingerprint"
        else:
            status_data["type"] = "unknown"
            status_data["name"] = "Unknown"

        print("The door will lock in 5 seconds!")
        time.sleep(5)
        
        face.clear_name()
        servo_control.lock()
        
        # Publish locked status
        status_data["state"] = 0
        status_data["time"] = time.time()
        publish_status(status_data)
        
        time.sleep(0.1)
        print("The door has been locked!")

except KeyboardInterrupt:
    print('Program interrupted. Cleaning up GPIO settings...')

finally:
    pubnub.unsubscribe().channels([CONTROL_CHANNEL]).execute()
    GPIO.cleanup()
