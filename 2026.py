
import cv2
import mediapipe as mp
import pygame
import numpy as np

# 初期設定
pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Harukaze 2026 - Hand Pose Game")
clock = pygame.time.Clock()

# MediaPipe Hand Landmarkerの準備
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# OpenCVカメラの準備
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# メインループ
running = True
while running:
    # イベント処理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        running = False

    # カメラからフレームを取得
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 画像を左右反転し、色をBGRからRGBに変換
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # MediaPipeで処理
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # 手のランドマークを描画
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    # OpenCVの画像をPygameのサーフェスに変換
    # 画像の次元を回転させてからNumpy配列をPygameサーフェスに変換
    image = np.rot90(image)
    image = pygame.surfarray.make_surface(image)
    # Pygameの座標系に合わせて画像をフリップ
    image = pygame.transform.flip(image, True, False)


    # 画面に描画
    screen.blit(image, (0, 0))
    pygame.display.flip()

    # フレームレートの設定
    clock.tick(60)

# 終了処理
cap.release()
hands.close()
pygame.quit()
