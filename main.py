import numpy as np
import input_data
import cv2
import Digit_Recognizer


def main():
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

    d1 = Digit_Recognizer.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=1500, alpha=0.05,
                                   print_cost=True)
    w_LR = d1["w"]
    b_LR = d1["b"]

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
                ans1 = Digit_Recognizer.predict(w_LR, b_LR, newImage)

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Digit: " + str(ans1), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Digit Recognition", img)
        k = cv2.waitKey(10)
        if k == 27:
            break


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1

