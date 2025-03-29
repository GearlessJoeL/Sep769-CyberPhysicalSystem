import RPi.GPIO as GPIO
import time

# 引脚定义 (BOARD 模式下的引脚)
RED_PIN = 11    # 对应 BCM 编号 GPIO17
GREEN_PIN = 13  # 对应 BCM 编号 GPIO27
BLUE_PIN = 15   # 对应 BCM 编号 GPIO22


# 初始化 LED 引脚 (只允许 MainControl.py 调用)
def init_led():
    if GPIO.getmode() is None:
        GPIO.setmode(GPIO.BOARD)  # 确保设置为 BOARD 模式
    elif GPIO.getmode() == GPIO.BCM:
        raise RuntimeError('GPIO 模式与引脚定义不匹配 (当前为 BCM 模式)。')

    GPIO.setwarnings(False)

    # 设置引脚为输出模式并初始化为低电平
    GPIO.setup(RED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(GREEN_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BLUE_PIN, GPIO.OUT, initial=GPIO.LOW)
    #print("LED 模块初始化成功")


# 设置 LED 颜色
def set_color(r, g, b):
    init_led()
    GPIO.output(RED_PIN, r)
    GPIO.output(GREEN_PIN, g)
    GPIO.output(BLUE_PIN, b)


def led_success():
    set_color(0, 1, 0)


def led_failure():
    set_color(1, 0, 0)


def led_waiting():
    set_color(1, 1, 0)


def turn_off_led():
    set_color(0, 0, 0)
