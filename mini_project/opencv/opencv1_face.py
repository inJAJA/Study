import cv2, dlib, sys
import numpy as  np

scaler = 0.2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/data/opencv/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('D:/data/opencv/face.mp4')

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    ori = img.copy()

    faces = detector(img)     # left, top, right, bottom 총 4개의 값 반환
    # print(faces)            # rectangles[[(89, 319) (192, 423)]] 
                              # [(face.left(), face.top()) (face.right(), face.bottom())]
    for face in faces:
        img = cv2.rectangle(img, pt1 = (face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 0),
                            thickness=2, lineType=cv2.LINE_AA)
    
    cv2.imshow('img', img)
    cv2.waitKey(1)

'''
# dlib install
1. activate base(가상환경 이름)
2. conda update --all
3. conda install -c conda-forge dlib
'''