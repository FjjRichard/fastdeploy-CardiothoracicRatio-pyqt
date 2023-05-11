# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/05/09 21:07:58
@Author  :   吖查 
@Version :   1.0
@Contact :   527055685@qq.com
'''

# here put the import lib


from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication
import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import fastdeploy.vision as vision
import numpy as np

from datetime import datetime

class InitModelThread(QThread):  
    """
    Initial model
    """
    signal_init_model = pyqtSignal(object)  
    def __init__(self, model_path=None):
        super(InitModelThread, self).__init__()
        self.model_path = model_path
        

    def run(self):
        model = vision.segmentation.PaddleSegModel(r'./unetmodel/model.pdmodel',
                                                        r'./unetmodel/model.pdiparams',
                                                        r'./unetmodel/deploy.yaml')
        self.signal_init_model.emit(model)



class MyApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui= uic.loadUi('myui.ui', self)

        self.rate = 0.00

        self.setWindowFlags(Qt.WindowCloseButtonHint|Qt.WindowStaysOnTopHint)

        self.model= object()
        self.clipboard = QtWidgets.QApplication.clipboard()
        self.clipboard.clear()#clear clipboard

        self.init_model_thread = InitModelThread()

        self.init_model_thread.signal_init_model.connect(self.initModel)
        self.init_model_thread.start()
        self.ui.btInfer.clicked.connect(self.infer)    

        
        self.ui.radioButton.clicked.connect(self.showPic)
        self.showpicWindow = PicWindow()

 
    def initModel(self,model):
        self.model = model

        self.init_model_thread.quit()

    def infer(self):
        """
        Reasoning the chest X-ray from the clipboard and calculating the cardiothoracic ratio
        """
        mdata = self.clipboard.mimeData()
        if mdata.hasImage():
            qimage = self.clipboard.image()
            im = self.qimageToCvimg(qimage)
            im = self.scalePic(im)

            h,w,_ = im.shape
            result = self.model.predict(im.copy())
            mask = np.array(result.label_map).reshape(h,w)

            vis_image , rate = self.caleRatio(im,mask)

            self.ui.value.setText(f"result:{str(rate)}")
            self.ui.radioButton.setChecked(False)

            if not os.path.exists("./preImg"):
                os.mkdir("./preImg")
            date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            self.ui.datetime.setText(f"{date}")
            cv2.imwrite(f"./preImg/vis_image.png", vis_image)
            self.clipboard.clear()


    def showPic(self):
        """
        Show images 
        """
        radio_status = self.ui.radioButton.isChecked()
        if radio_status:
            self.showpicWindow.showPic()
            self.showpicWindow.show()
        else:
            self.showpicWindow.hide()
            
    def qimageToCvimg(self,qimg):
        """
        Get a picture from the clipboard
        """
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)

        result = result[..., :3]
        return result

    def scalePic(self,im):
        """
        Scale the shortest edge of the image to 512
        """
        h,w,_ = im.shape
        if h > w :
            scale = 512 / w
            scale_w = 512 
            scale_h = int(h * scale)
            im = cv2.resize(im, (scale_w, scale_h))
        else:
            scale = 512 / h
            scale_h = 512
            scale_w = int(w*scale)
            im = cv2.resize(im, (scale_w, scale_h))
        return im


    def caleRatio(self,image,output):
        """
        Calculate cardiothoracic ratio
        """
        h,w = output.shape[0], output.shape[1]
        segmentation = np.zeros((h,w),np.uint8)

        segmentation[:, :] = output[:,:] 

        #Find the location of the heart
        ret, heart = cv2.threshold(segmentation, 1, 255,0)
        heart = self.morph_open(heart)
        contours, hierarchy = cv2.findContours(heart, 1,2)
        max_idx = self.find_max_area(contours)
        heart = contours[max_idx]
        #Obtain the leftmost and rightmost points of the heart, as well as the x, y, width, height of the bounding rectangle
        left_heart, right_heart ,hx,hy,WidthHeart,hh= self.get_left_right(heart)

        #Find the location of the lungs
        img_temp = segmentation.copy()
        h,w = img_temp.shape[0], img_temp.shape[1]
        #Treating the heart and lungs as a whole facilitates the calculation of boundary rectangles
        ret, chest = cv2.threshold(img_temp, 0, 255,0)
        chest = self.morph_open(chest)
        contours, hierarchy = cv2.findContours(chest, 1,2)
        max_idx = self.find_max_area(contours)
        chest = contours[max_idx]
        #Obtain the leftmost and rightmost points of the lungs, as well as the x, y, width, height of the outer margin
        left_chest, right_chest,cx,cy,WidthChest,ch = self.get_left_right(chest)

        #Draw the maximum transverse diameter of the lungs and the transverse diameter of the heart
        cv2.line(image,left_heart,(left_heart[0] + int(WidthHeart/2),left_heart[1]),(0,255,0),2)
        cv2.line(image,right_heart,(right_heart[0] - int(WidthHeart/2),right_heart[1]),(0,255,0),2)
        cv2.line(image,left_chest,(left_chest[0] + WidthChest,left_chest[1]),(0,0,255),2)

        #Calculate cardiothoracic ratio
        rate = round(WidthHeart/WidthChest,3)
        return image, rate


    def find_max_area(self,contours):
        # Find the largest contour
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))
        return max_idx

    def morph_open(self,threshold):
        #Open operation to remove noise
        kernel = np.ones((3,3), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,kernel, iterations=1)
        return threshold

    def get_left_right(self,cnt):
        #Calculate the x, y, w, h of the outer margin for the leftmost and last points
        left = tuple(cnt[cnt[:,:,0].argmin()][0])
        right = tuple(cnt[cnt[:,:,0].argmax()][0])
        x,y,w,h = cv2.boundingRect(cnt)
        return (left, right,x,y,w,h)
    
class PicWindow(QtWidgets.QWidget):
    """
    Show inference images
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inference images")
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.label = QLabel()
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        self.setLayout(vbox)


    def showPic(self):
        if os.path.exists("./preImg/vis_image.png"):
            self.label.setPixmap(QPixmap("./preImg/vis_image.png"))	
        else:
            self.label.setText("None images")

 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    sys.exit(app.exec_())