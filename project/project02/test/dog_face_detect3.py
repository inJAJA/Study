import dlib, cv2, os                            
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import time
dlib.DLIB_USE_CUDA = True
''' 견종별로 따로 따로 저장 
    label없음 ( 추가로 라벨 만들어 줘야 함 )'''


start = time.time()

train_path = 'D:/data/project/face'


def face_detector(path, folder, w, h):
    # path 안에 있는 폴더를 지정하여 해당 폴더의 이미지에만 적용 

    print('---------- START %s ---------'%(folder))
    image_dir = path + '/'+folder+'/'
    X = []

    f = os.listdir(image_dir)                       # 폴더내 파일 이름 찾기
    print(len(f))

    # f = f[:100]
    # print(len(f))
    for filename in f:                              # 파일 별로 이미지 불러오기
        img = cv2.imread(image_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv는 BGR로 불러 들임으로 볼 때 우리가 원하는 색으로 보기 위해 RGB로

        # img_result = img.copy()                   # 원본 이미지 copy

        detector = dlib.cnn_face_detection_model_v1('./project/project02/weight/dogHeadDetector.dat')
        dets = detector(img, upsample_num_times=1)

        x = X.append

        img_result = img.copy()

        for i, d in enumerate(dets):

            x1, y1 = d.rect.left(), d.rect.top()
            x2, y2 = d.rect.right(), d.rect.bottom()
            pad = (x2 - x1)

            x1 = x1 - pad/4
            y1 = y1 - pad*3/8
            x2 = x2 + pad/4
            y2 = y2 + pad/8

            x1, x2, y1, y2 = map(int, (x1, x2, y1, y2)) # int형으로 변환

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            
            # boundong box 보여주기
            # cv2.rectangle(img_result, (x1, y1), (x2, y2), thickness=1, color=(122, 122, 122), lineType=cv2.LINE_AA)

            # plt.figure(figsize=(16, 16))
            # plt.imshow(img_result)
            # plt.show()

            cropping = img[y1:y2, x1:x2]
            crop = cv2.resize(cropping, dsize = (w, h), interpolation = cv2.INTER_LINEAR)

            # 잘린 이미지 보기
            # plt.figure(figsize=(16, 16))
            # plt.imshow(crop)
            # plt.show()


            x(crop/255)
        
    images = np.array(X)

    np.save('./project/project02/data/face/face_image_%s.npy'%(folder), images)     # face_image_{folder} 이름으로 저장

    print('---------- END %s ---------'%(folder))

    
face_detector(train_path, 'Dachshund', 128, 128)




