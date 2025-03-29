import RPi.GPIO as GPIO
import time

# 引脚定义
BUZZER_PIN = 31  # 对应于 GPIO6 (在 BOARD 模式下的 Pin 31)

buzzer_initialized = False  # 引脚初始化状态变量


def init_buzzer():
    global buzzer_initialized

    if not buzzer_initialized:  # 确保只初始化一次
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
        elif GPIO.getmode() == GPIO.BCM:
            raise RuntimeError('GPIO 模式已被设置为 BCM，与 BOARD 不兼容。')

        GPIO.setwarnings(False)
        GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.HIGH)
        buzzer_initialized = True
        print("蜂鸣器初始化成功")


def buzzer_success():
    init_buzzer()
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.cleanup(BUZZER_PIN)


def buzzer_failure():
    init_buzzer()
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.cleanup(BUZZER_PIN)


def cleanup():
    GPIO.cleanup()
