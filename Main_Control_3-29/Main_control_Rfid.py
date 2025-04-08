import time
import RPi.GPIO as GPIO
import rfid_reader_v2 as rfid_reader
import led_control_v2 as led_control
import buzzer_control_v3 as buzzer_control
import servo_control_v2 as servo_control
import fingerprint_reader_v2 as fingerprint_reader
import threading
from rfid_reader_v2 import exit_event
from face_recog import FaceRecognition
from pubnub.callbacks import SubscribeCallback
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
import queue
import json
import os
from dotenv import load_dotenv


load_dotenv()
# PubNub Configuration
pnconfig = PNConfiguration()
pnconfig.subscribe_key = os.getenv('PYTHON_PUBNUB_SUBSCRIBE_KEY')
pnconfig.publish_key = os.getenv('PYTHON_PUBNUB_PUBLISH_KEY')
pnconfig.uuid = os.getenv('PYTHON_PUBNUB_UUID')
pubnub = PubNub(pnconfig)

# Channel names
CONTROL_CHANNEL = os.getenv('PYTHON_PUBNUB_CONTROL_CHANNEL')
STATUS_CHANNEL =  os.getenv('PYTHON_PUBNUB_STATUS_CHANNEL')

# Initialize GPIO mode
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# Initialize modules
face = FaceRecognition()

rfid_success = False
face_success = False
fingerprint_success = False
remote_unlock = False

class MySubscribeCallback(SubscribeCallback):
    def message(self, pubnub, message):
        global remote_unlock
        if message.channel == CONTROL_CHANNEL:
            if message.message.get('message_type') == "control" and message.message.get('action') == 'unlock':
                remote_unlock = True
                print("message received")

                
#                 # Publish status update
#                 status_data = {
#                     "message_type": "status",
#                     "state": 1,
#                     "type": "remote",
#                     "time": time.time(),
#                     "name": "Remote Access"
#                 }
#                 publish_status(status_data)
#                 
#                 # Auto-lock after 5 seconds
#                 time.sleep(5)
#                 servo_control.lock()
#                 
#                 # Update locked status
#                 status_data["state"] = 0
#                 status_data["time"] = time.time()
#                 publish_status(status_data)

pubnub.add_listener(MySubscribeCallback())
pubnub.subscribe().channels([CONTROL_CHANNEL]).execute()

def publish_status(status_data):
    status_data["message_type"] = "status"  # Add message type
    pubnub.publish().channel(STATUS_CHANNEL).message(status_data).sync()

def rfid_authentication():
    global rfid_success
    while not rfid_success and not face_success and not remote_unlock and not fingerprint_success:
        if face_success or remote_unlock or fingerprint_success:
            print("Face or fingerprint recognition succeeded or remote unlock activated, RFID thread exiting.")
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


def remote_authentication():
    global remote_unlock
    while not rfid_success and not face_success and not remote_unlock:
        if remote_unlock == True:
            print("Received remote unlock command")
            led_control.led_success()
            buzzer_control.buzzer_success()
            servo_control.unlock()
            exit_event.set()
            return
        


def face_authentication():
    global face_success
    print('please face the camera...')
    while not rfid_success and not face_success and not remote_unlock and not fingerprint_success:
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

def fingerprint_authentication():
    global fingerprint_success
    while not fingerprint_success and not rfid_success and not face_success and not remote_unlock: # 循环等待验证
        if fingerprint_success:
            print("检测到指纹识别成功，指纹线程自动退出。")
            exit_event.set()
            return
        
        led_control.led_waiting()
        fingerprint_status = fingerprint_reader.get_fingerprint_detail()

        if fingerprint_status:
            print('指纹验证成功')
            led_control.led_success()
            buzzer_control.buzzer_success()
            servo_control.unlock()
            fingerprint_success = True
            return

try:
    rfid_thread = threading.Thread(target=rfid_authentication)
    face_thread = threading.Thread(target=face_authentication)
    remote_thread = threading.Thread(target=remote_authentication)

    rfid_thread.start()
    face_thread.start()
    remote_thread.start()

    rfid_thread.join()
    face_thread.join()
    remote_thread.join()
#     print("exed")
    if face_success or rfid_success or remote_unlock: #or fingerprint_success
        status_data = {
            "state": 1,
            "type": "",
            "time": time.time(),
            "name": ""
        }

        if face_success:
            status_data["type"] = "face"
            status_data["name"] = face.get_name()
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
