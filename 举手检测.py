import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import random
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import time
from tqdm import tqdm, trange

window = tk.Tk()
window.title("举手检测")

canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

video_sign = False
img_path = None
video_path = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)


def clearAll():
    canvas.delete(tk.ALL)


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle


def get_pos(keypoints):
    str_pose = ""
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


def process_video(video_path):
    video_flag = False
    cap = cv2.VideoCapture(video_path)
    out_path = "./out_Data.mp4"
    print("视频开始处理……")
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print("总帧数 = ", frame_count)
    cap = cv2.VideoCapture(video_path)
    if video_flag == False:
        frame_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 处理图像的尺寸。
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])), )  # 输出图像的句柄
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    pbar.update(1)
                    frame = process_frame(frame)
                    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                    cv2.imshow("frame", frame)
                    out.write(frame)
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    break
        except:
            print("中途中断")
            pass
    cap.release()
    cv2.destroyAllWindows()
    out.release()
    print("视频已保存至", out_path)


def pre_image(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = image.copy()
    frame = process_frame(img)
    return frame


def open_video():
    clearAll()
    file_path = filedialog.askopenfilename(filetypes=[("Vedio files", ".mp4")])
    if file_path:
        global video_path
        video_path = file_path
        global video_sign
        video_sign = True
        return video_sign


def show_video():
    if video_sign:
        process_video(video_path)


def open_image():
    clearAll()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg .png .bmp")])
    if file_path:
        image = Image.open(file_path).convert('RGB')
        width, height = image.size
        scale = min(800 / width, 600 / height)
        image = image.resize((int(width * scale), int(height * scale)))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(400, 300, image=photo)
        global img
        img = photo
        global img_path
        img_path = file_path
        return image


def show_keypoints():
    global img
    global img_path
    if img and img_path:
        image = np.array(img)
        image = pre_image(img_path)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        original_width = img.width()
        original_height = img.height()
        processed_width = image.width
        processed_height = image.height
        image = image.resize((original_width, original_height))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(400, 300, image=photo)
        img = photo


open_button = tk.Button(window, text="打开图片", command=open_image)
open_button.pack(side=tk.LEFT)

keypoints_button = tk.Button(window, text="检测结果", command=show_keypoints)
keypoints_button.pack(side=tk.LEFT)

open_button = tk.Button(window, text="打开视频", command=open_video)
open_button.pack(side=tk.RIGHT)

open_button = tk.Button(window, text="检测视频", command=show_video)
open_button.pack(side=tk.RIGHT)

window.mainloop()
