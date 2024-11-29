import cv2  # 导入 OpenCV 库，用于图像处理和计算机视觉
import mediapipe as mp  # 导入 MediaPipe 库，用于手势识别
import time  # 导入时间库，用于计算帧率
from gesture_judgment import detect_all_finger_state, detect_hand_state  # 导入手势判断相关的函数

# 初始化 MediaPipe 的手部检测模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # 用于绘制手部关键点的工具

# 存储最近 30 帧的手势判断结果
recent_states = [''] * 30

# 打开摄像头
cap = cv2.VideoCapture(0)

prev_time = 0  # 用于计算帧率
while True:
    ret, frame = cap.read()  # 从摄像头读取一帧
    frame = cv2.flip(frame, 1)  # 水平翻转图像，使其看起来像镜子
    h, w = frame.shape[:2]  # 获取图像的高度和宽度
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB 格式
    keypoints = hands.process(image)  # 处理图像，检测手部关键点

    if keypoints.multi_hand_landmarks:  # 如果检测到手部关键点
        lm = keypoints.multi_hand_landmarks[0]  # 获取第一个手的关键点
        lmHand = mp_hands.HandLandmark  # 获取手部关键点的枚举类

        # 创建一个列表来存储每根手指的关键点坐标
        landmark_list = [[] for _ in range(6)]  # 6 个子列表：根节点和 5 根手指

        # 遍历手部关键点，提取坐标
        for index, landmark in enumerate(lm.landmark):
            x = int(landmark.x * w)  # 将相对坐标转换为绝对像素坐标
            y = int(landmark.y * h)
            if index == lmHand.WRIST:  # 第 0 个关键点是手腕
                landmark_list[0].append((x, y))
            elif 1 <= index <= 4:  # 第 1 到 4 个关键点是拇指
                landmark_list[1].append((x, y))
            elif 5 <= index <= 8:  # 第 5 到 8 个关键点是食指
                landmark_list[2].append((x, y))
            elif 9 <= index <= 12:  # 第 9 到 12 个关键点是中指
                landmark_list[3].append((x, y))
            elif 13 <= index <= 16:  # 第 13 到 16 个关键点是无名指
                landmark_list[4].append((x, y))
            elif 17 <= index <= 20:  # 第 17 到 20 个关键点是小指
                landmark_list[5].append((x, y))

        # 获取所有关节点的坐标
        point0 = landmark_list[0][0]  # 手腕坐标
        point1, point2, point3, point4 = landmark_list[1]  # 拇指的 4 个关键点
        point5, point6, point7, point8 = landmark_list[2]  # 食指的 4 个关键点
        point9, point10, point11, point12 = landmark_list[3]  # 中指的 4 个关键点
        point13, point14, point15, point16 = landmark_list[4]  # 无名指的 4 个关键点
        point17, point18, point19, point20 = landmark_list[5]  # 小指的 4 个关键点

        # 将所有关键点的坐标存储到字典中，简化后续函数的参数
        all_points = {
            'point0': landmark_list[0][0],
            'point1': landmark_list[1][0], 'point2': landmark_list[1][1], 'point3': landmark_list[1][2], 'point4': landmark_list[1][3],
            'point5': landmark_list[2][0], 'point6': landmark_list[2][1], 'point7': landmark_list[2][2], 'point8': landmark_list[2][3],
            'point9': landmark_list[3][0], 'point10': landmark_list[3][1], 'point11': landmark_list[3][2], 'point12': landmark_list[3][3],
            'point13': landmark_list[4][0], 'point14': landmark_list[4][1], 'point15': landmark_list[4][2], 'point16': landmark_list[4][3],
            'point17': landmark_list[5][0], 'point18': landmark_list[5][1], 'point19': landmark_list[5][2], 'point20': landmark_list[5][3]
        }

        # 调用函数，判断每根手指的弯曲或伸直状态
        bend_states, straighten_states = detect_all_finger_state(all_points)

        # 调用函数，检测当前手势
        current_state = detect_hand_state(all_points, bend_states, straighten_states)

        # 更新最近状态列表
        recent_states.pop(0)  # 移除最旧的状态
        recent_states.append(current_state)  # 添加当前状态

        # 检查列表中的所有状态是否相同
        if len(set(recent_states)) == 1:  # 如果最近 30 帧的手势状态都相同
            print("Detected consistent hand state:", recent_states[0])  # 输出当前手势
            cv2.putText(frame, current_state, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # 在图像上显示手势状态

        # 绘制手部关键点和连接线
        for hand_landmarks in keypoints.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    curr_time = time.time()  # 获取当前时间
    fps = 1 / (curr_time - prev_time)  # 计算帧率
    prev_time = curr_time  # 更新上一次时间

    # 在画面上绘制帧率
    # 使用 cv2.putText 函数在图像上绘制当前帧率 (FPS)
    # 参数解释：
    # frame: 要在其上绘制文本的图像
    # f"FPS: {int(fps)}": 格式化字符串，显示帧率的整数值
    # (10, 30): 文本的位置，左上角为 (10, 30)
    # cv2.FONT_HERSHEY_SIMPLEX: 文本字体
    # 1: 文本大小
    # (255, 255, 255): 文本颜色，这里为白色
    # 2: 文本的粗细
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示带有手部检测的图像
    cv2.imshow("Hand Detection", frame)
    # 使用 cv2.imshow 函数创建一个窗口，显示标题为 "Hand Detection" 的图像窗口，并在其中显示当前的 frame

    # 检查用户是否按下 "q" 键
    if cv2.waitKey(1) == ord("q"):
        # cv2.waitKey(1): 等待 1 毫秒并检测键盘输入，如果按下 "q" 键，将返回按键的 ASCII 码
        break  # 如果检测到 "q" 键，跳出循环，结束程序

# 释放摄像头资源
cap.release()
# 调用 cap.release() 释放摄像头资源，关闭摄像头

# 关闭所有 OpenCV 创建的窗口
cv2.destroyAllWindows()
# 调用 cv2.destroyAllWindows() 关闭所有由 OpenCV 创建的窗口
