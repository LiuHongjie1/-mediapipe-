import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import random
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# 创建一个Tkinter窗口
window = tk.Tk()
window.title("举手检测")

# 创建一个画布，用于显示图片
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# 定义一个全局变量，用于保存图片文件的路径
img_path = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle


def get_pos(keypoints):
    str_pose = ""
    # 计算左臂与水平方向的夹角
    keypoints = np.array(keypoints)
    p_x_right, p_y_right = keypoints[16]
    p_x_left, p_y_left = keypoints[15]
    p_x_right_body, p_y_right_body = keypoints[11]
    p_x_left_body, p_y_left_body = keypoints[12]

    if p_y_right < p_y_right_body:
        v1 = keypoints[14] - keypoints[16]
        v2 = keypoints[12] - keypoints[11]
        get_right_arm = abs(get_angle(v1, v2))
        # print("R{}".format(get_right_arm))
        if 60 < get_right_arm < 120:
            str_pose = "RIGHT RAISE"
    elif p_y_left < p_y_left_body:
        v1 = keypoints[13] - keypoints[15]
        v2 = keypoints[11] - keypoints[12]
        get_left_arm = abs(get_angle(v1, v2))
        # print("L{}".format(get_left_arm))
        if 60 < get_left_arm < 120:
            str_pose = "LEFT RAISE"
    else:
        str_pose = "NO RAISE"

    return str_pose


def drawImage(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('simsun.ttc', 200, encoding="utf-8")
    fillColor = color
    position = pos
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, fillColor, font)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def process_frame(img):
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    tl = round(0.005 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    keypoints = ['' for i in range(33)]
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            keypoints[i] = (cx, cy)
    else:
        print("NO PERSON")
        struction = "NO PERSON"
        img = cv2.putText(img, struction, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0),
                          6)
    end_time = time.time()
    process_time = end_time - start_time
    fps = 1 / process_time
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(33)]
    radius = [random.randint(8, 15) for _ in range(33)]
    for i in range(33):
        cx, cy = keypoints[i]
        img = cv2.circle(img, (cx, cy), radius[i], colors[i], -1)
        str_pose = get_pos(keypoints)
    cv2.putText(img, "{}".format(str_pose), (0, 500), cv2.FONT_HERSHEY_TRIPLEX,
                tl / 3, (0, 255, 0), thickness=tf)
    return img


def pre_image(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = image.copy()
    frame = process_frame(img)
    return frame


# 定义一个函数，用于打开图片文件
def open_image():
    # 使用filedialog模块选择图片文件
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg .png .bmp")])

    # 如果文件路径不为空，打开图片文件
    if file_path:
        # 使用PIL模块读取图片文件
        image = Image.open(file_path).convert('RGB')
        # 获取图片的宽度和高度
        width, height = image.size
        # 计算图片的缩放比例，使其适应画布的大小
        scale = min(800 / width, 600 / height)
        # 使用PIL模块缩放图片
        image = image.resize((int(width * scale), int(height * scale)))
        # 使用PIL模块将图片转换为Tkinter兼容的格式
        photo = ImageTk.PhotoImage(image)
        # 在画布上显示图片
        canvas.create_image(400, 300, image=photo)
        # 将图片对象保存为全局变量，防止被垃圾回收
        global img
        img = photo
        # 将图片文件的路径保存为全局变量，方便后续处理
        global img_path
        img_path = file_path
        # 返回图片对象
        return image


# 定义一个函数，用于显示关键点
def show_keypoints():
    # 获取当前画布上的图片对象
    global img
    # 获取当前图片文件的路径
    global img_path
    if img and img_path:
        # 使用PIL模块将图片对象转换为numpy数组
        image = np.array(img)
        # 调用mediapipe_reshape.py文件中的process_frame(img_path)函数，返回处理后的图片
        image = pre_image(img_path)
        # 使用PIL模块将numpy数组转换为图片对象
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 获取原图片的宽度和 高度
        original_width = img.width()
        original_height = img.height()
        # 获取处理后的图片的宽度和高度
        processed_width = image.width
        processed_height = image.height
        # 使用PIL模块将图片对象缩放到和原图片一样的大小
        image = image.resize((original_width, original_height))
        # 使用PIL模块将图片对象转换为Tkinter兼容的格式
        photo = ImageTk.PhotoImage(image)
        # 在画布上显示图片
        canvas.create_image(400, 300, image=photo)
        # 将图片对象保存为全局变量，防止被垃圾回收
        img = photo


# 创建一个按钮，用于打开图片文件
open_button = tk.Button(window, text="打开图片", command=open_image)
open_button.pack(side=tk.LEFT)

# 创建一个按钮，用于显示关键点
keypoints_button = tk.Button(window, text="检测结果", command=show_keypoints)
keypoints_button.pack(side=tk.RIGHT)

# 进入Tkinter主循环
window.mainloop()
