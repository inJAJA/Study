import numpy as np
import cv2

def FAST(img_path):
    img = cv2.imread(img_path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2, img3 = None, None

    fast = cv2.FastFeatureDetector_create(100)              # FAST 적용
                                                            # /threshold = 100(값이 커질수록 검출하는 keypoint개수 줄어듬)
    kp = fast.detect(img, None)                             # keypoint 검출
    img2 = cv2.drawKeypoints(img, kp, img2, (255, 0, 0))    # keypoint 그리기
    cv2.imshow('FAST1', img2)

    fast.setNonmaxSuppression(0)                            # 0 = Non-maximal-Suppression을 False로 두고 keypoint검출/ default = True
    kp = fast.detect(img, None)
    img3 = cv2.drawKeypoints(img, kp, img3, (255, 0, 0))
    cv2.imshow('FAST2', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = './project/project02/data/dog2.jpg'
FAST(img_path)