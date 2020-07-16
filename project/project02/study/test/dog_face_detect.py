import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

img_path = './project/project02/data/dog2.jpg'

def face_detector(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_result = img.copy()

    detector = dlib.cnn_face_detection_model_v1('./project/project02/weight/dogHeadDetector.dat')
    dets = detector(img, upsample_num_times=1)

    for i, d in enumerate(dets):
        print("Detection {}: Left:{} Top:{} Right:{} Bottom:{} Confidence:{}".format(i, 
                d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        s = 30
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()

        cv2.rectangle(img_result, (x1, y1-s), (x2+s, y2+s), thickness=2, color=(122, 122, 122), lineType=cv2.LINE_AA)

    # plt.figure(figsize=(16, 16))
    # plt.imshow(img_result)
    # plt.show()
    return dets

def shape_detector(img_path, dets): # image경로와 face_detector로 검출한 bbox필요
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_result = img.copy()

    shapes = []

    predictor = dlib.shape_predictor('./project/project02/weight/landmarkDetector.dat')

    for i, d in  enumerate(dets):
        shape = predictor(img, d.rect)
        shape = face_utils.shape_to_np(shape)

        for i, p in enumerate(shape):
            shapes.append(shape)
            cv2.circle(img_result, center=tuple(p), radius=3, color=(0,0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    img_out = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(16, 16))
    plt.imshow(img_result)
    plt.show()

    return shape

face = face_detector(img_path)
print(face)                     # <dlib.mmod_rectangles object at 0x000001D08439C870>

shape = shape_detector(img_path, face)
print(shape)                    # 특징점 좌표


