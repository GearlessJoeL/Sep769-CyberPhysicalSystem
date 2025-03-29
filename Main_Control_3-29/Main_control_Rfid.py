import time
import RPi.GPIO as GPIO
import rfid_reader_v2 as rfid_reader
import led_control_v2 as led_control
import buzzer_control_v3 as buzzer_control
import servo_control_v2 as servo_control
import threading
from rfid_reader_v2 import exit_event  # 从 rfid_reader 模块导入退出事件
from face_recog import FaceRecognition

# 初始化 GPIO 模式
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# 初始化模块
face = FaceRecognition()

rfid_success = False
face_success = False


def rfid_authentication():
    global rfid_success
    while not rfid_success and not face_success:  # 循环等待验证
        if face_success:  # 如果人脸识别成功，退出 RFID 线程
            print("检测到人脸识别成功，RFID 线程自动退出。")
            exit_event.set()  # 设置退出事件
            return

        led_control.led_waiting()
        card_id = rfid_reader.read_rfid()

        if card_id == '860338929300':  # 假定这是授权卡片
            print('RFID 验证成功')
            led_control.led_success()
            buzzer_control.buzzer_success()
            servo_control.unlock()
            rfid_success = True
            return  # 停止线程

def face_authentication():
    global face_success
    print('please face the camera...')
    while not rfid_success and not face_success:  # 循环等待验证
        face.recognize()
        if face.name != "":
            led_control.led_success()
            buzzer_control.buzzer_success()
            servo_control.unlock()
            face_success = True  # 更新全局变量
            exit_event.set()  # 通知 RFID 线程退出                
            return  # 确保人脸识别线程退出


try:
    # 创建两个线程同时运行 RFID 和 人脸识别
    rfid_thread = threading.Thread(target=rfid_authentication)
    face_thread = threading.Thread(target=face_authentication)

    rfid_thread.start()
    face_thread.start()

    rfid_thread.join()
    face_thread.join()
    
    if face_success or rfid_success:
        print("The door will lock in 5 seconds!")
        print(face.name)
        time.sleep(5)
        face.clear_name()
        servo_control.lock()
        time.sleep(0.1)
        print("The door has been locked!")


except KeyboardInterrupt:
    print('程序已中断。正在清理 GPIO 设置...')

finally:
    GPIO.cleanup()
