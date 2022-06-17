import cv2
from tkinter import *
from PIL import Image, ImageTk, ImageStat
import numpy as np
from skimage.feature import hog
import math
import input_data
import Digit_Recognizer
import main
cam_on = False
cap = None
mainWindow = Tk()

mainFrame = Frame(mainWindow, height=640, width=1200)
mainFrame.place(x=350, y=0)
mainWindow['bg'] = '#49A'

cameraFrame = Frame(mainWindow, height=800, width=305)
cameraFrame.place(x=0, y=0)
cameraFrame['bg'] = '#49A'
def brightness(im_file):
   stat = ImageStat.Stat(im_file)
   gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
         for r, g, b in im_file.getdata())
   return sum(gs)/stat.count[0]


def show_frame():
    if cam_on:
        ret, frame = cap.read()

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)
            vid_lbl.imgtk = imgtk
            vid_lbl.configure(image=imgtk)

        vid_lbl.after(10, show_frame)


def show_frame_FILTER():
    if cam_on:
        ret, frame = cap.read()

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((1100, 700))
            cv2filterImage = [];
            if 0 < brightness(img) <= 51:
                cv2filterImage = Roberts(cv2image)
                print("Roberts")
            elif 52 < brightness(img) <= 102:
                cv2filterImage = Sobel(cv2image)
                print("Sobel")
            elif 103 < brightness(img) <= 153:
                cv2filterImage = Prewitt(cv2image)
                print("Prewitt")
            elif 154 < brightness(img) <= 204:
                cv2filterImage = Canny(cv2image)
                print("Canny")
            elif 205 < brightness(img) <= 255:
                cv2filterImage = Laplacian(cv2image)
                print("Laplacian")
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)    
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_FILTER)

def show_frame_roberts():
    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            cv2filterImage = Roberts(cv2image)
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_roberts)

def show_frame_canny():
    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            cv2filterImage = Canny(cv2image)
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_canny)

def show_frame_roberts():
    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            cv2filterImage = Roberts(cv2image)
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_roberts)

def show_frame_prewitt():
    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            cv2filterImage = Prewitt(cv2image)
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_prewitt)

def show_frame_laplacian():
    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            cv2filterImage = Laplacian(cv2image)
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_laplacian)

def show_frame_sobel():
    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            cv2filterImage = Sobel(cv2image)
            img = Image.fromarray(cv2filterImage).resize((1100, 700))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        vid_lbl.after(10, show_frame_sobel)

def start_vid():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame()

def start_vid_recognize():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_recognize()

def start_video_filter():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_FILTER()

def stop_vid():
    global cam_on
    cam_on = False
    
    if cap:
        cap.release()

def canny_method():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_canny()

def roberts_method():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_roberts()

def prewitt_method():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_prewitt()

def laplacian_method():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_laplacian()

def sobel_method():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0) 
    show_frame_sobel()

def Canny(image): 
    canny = cv2.Canny(image, 50, 240)
    return canny

def Roberts(image):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    y = cv2.filter2D(image, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    Roberts1 = cv2.resize(Roberts, (800, 400))
    return Roberts1

def Prewitt(image): 
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    y = cv2.filter2D(image, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    Prewitt1 = cv2.resize(Prewitt, (1600, 900))
    return Prewitt1

def Laplacian(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplace = np.uint8(np.absolute(laplace))
    return laplace

def Sobel(image):
    sobelx64f = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u

def show_frame_recognize():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    data = mnist.train.next_batch(8000)
    train_x = data[0]
    Y = data[1]
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    tb = mnist.train.next_batch(2000)
    Y_test = tb[1]
    X_test = tb[0]
    # 0.00002-92
    # 0.000005-92, 93 when 200000 190500

    # d1 = Digit_Recognizer_LR.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=1500, alpha=0.05,
    #                                print_cost=True)
    # w_LR = d1["w"]
    # b_LR = d1["b"]

    d1 = Digit_Recognizer.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=1500, alpha=0.05,
                                   print_cost=True)
 
    w_LR = d1["w"]
    b_LR = d1["b"]
    # dims = [784, 100, 80, 50, 10]
    # d3 = Digit_Recognizer_DL.model_DL(train_x.T, train_y.T, Y, X_test.T, Y_test, dims, alpha=0.5, num_iterations=1100,
    #                                   print_cost=True)

    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''
        ans2 = ''
        ans3 = ''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                # print(predict(w_from_model,b_from_model,contour))
                x, y, w, h = cv2.boundingRect(contour)
                # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage = np.array(newImage)
                newImage = newImage.flatten()
                newImage = newImage.reshape(newImage.shape[0], 1)
                # ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
                ans1 = Digit_Recognizer.predict(w_LR, b_LR, newImage)
                # ans3 = Digit_Recognizer_DL.predict(d3, newImage)

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Digit: " + str(ans1), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.imshow("Frame", img)


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


vid_lbl = Label(mainFrame)
vid_lbl.grid(row=0, column=0)

# Buttons
TurnCameraOn = Button(cameraFrame, text="Start Video", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=start_vid)
TurnCameraOn.place(x = 20, y = 10)
TurnCameraOn = Button(cameraFrame, text="Roberts Filter", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=roberts_method)
TurnCameraOn.place(x = 20, y = 90)
TurnCameraOn = Button(cameraFrame, text="Prewitt Filter", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=prewitt_method)
TurnCameraOn.place(x = 20, y = 170)
TurnCameraOn = Button(cameraFrame, text="Laplacian Filter", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=laplacian_method)
TurnCameraOn.place(x = 20, y = 250)
TurnCameraOn = Button(cameraFrame, text="Sobel Filter", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=sobel_method)
TurnCameraOn.place(x = 20, y = 330)
TurnCameraOn = Button(cameraFrame, text="Canny Filter", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=canny_method)
TurnCameraOn.place(x = 20, y = 410)
TurnCameraOff = Button(cameraFrame, text="Stop Video", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=stop_vid)
TurnCameraOff.place(x = 20, y = 490)
TurnCameraOff = Button(cameraFrame, text="Choose Filter", background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=start_video_filter)
TurnCameraOff.place(x = 20, y = 570)
TurnCameraOff = Button(cameraFrame, text="Recognize Digit",background="#555", foreground="#ccc",
  padx="20", pady="8", font="16", width = 20, command=main.main)
TurnCameraOff.place(x = 20, y = 650)

mainWindow.mainloop()