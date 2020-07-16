import cv2
import numpy as np
import panel as pn
from panel.widgets import IntSlider
from ipywidgets import AppLayout, Image, Layout, Box, widgets  # pip install ipywidgets
from IPython import display

img_path = './project/project02/data/dog2.jpg'
src = cv2.imread(img_path)
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#Component 선언
IntSlider_Threshold = IntSlider(
    value=1,
    min=1,
    max=50,
    step=1,
    description='Threshold: ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

def layout(header, left, right):
    layout = AppLayout(header=header,
          left_sidebar=left,
          center=None,
          right_sidebar=right)
    return layout

wImg_original = Image(layout = Layout(border="solid"), width="45%")
wImg_dst = Image(layout = Layout(border="solid"), width="45%")
wImg_dst2 = Image(layout = Layout(border="solid"), width="45%")
wImg_dst3 = Image(layout = Layout(border="solid"), width="45%")

items = [wImg_original,wImg_dst]
items2 = [wImg_dst2,wImg_dst3]
Left_image =Box(items)
Right_image = Box(items2)
box = layout(IntSlider_Threshold,Left_image,Right_image)

tab_nest = widgets.Tab()
tab_nest.children = [box]
tab_nest.set_title(0, 'Fast Feature Detect1')
tab_nest

tmpStream = cv2.imencode(".jpeg", src)[1].tostring()
wImg_original.value = tmpStream

display.display(tab_nest)

#Event 선언
threshold = 1
    
def on_value_change_Threshold(change):
    global threshold
    threshold = change['new']
    make_fastfeature(threshold)

def make_fastfeature(input_threshold):
    fastF = cv2.FastFeatureDetector.create(threshold = input_threshold)
    kp = fastF.detect(gray)
    dst = cv2.drawKeypoints(gray,kp,None,color=(0,0,255))
    print('len(kp)=',len(kp))

    fastF.setNonmaxSuppression(False)
    kp2 = fastF.detect(gray)
    dst2 = cv2.drawKeypoints(src,kp2,None,color=(0,0,255))
    print('len(kp2)=',len(kp2))

    dst3 = src.copy()
    points = cv2.KeyPoint_convert(kp)
    for cx,cy in points:
        cv2.circle(dst3,(cx,cy),3,color=(255,0,0),thickness=-1)
        
    tmpStream = cv2.imencode(".jpeg", dst)[1].tostring()
    wImg_dst.value = tmpStream
    
    tmpStream = cv2.imencode(".jpeg", dst2)[1].tostring()
    wImg_dst2.value = tmpStream
    
    tmpStream = cv2.imencode(".jpeg", dst3)[1].tostring()
    wImg_dst3.value = tmpStream

#초기화 작업
make_fastfeature(threshold)

#Component에 Event 장착
IntSlider_Threshold.observe(on_value_change_Threshold, names='value')