import cv2


def face_recognization(test_img):
    try:
        grayImg = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        face_haar_cascade = cv2.CascadeClassifier(
            "./haarcascade/haarcascade_frontalface_default.xml")
        face = face_haar_cascade.detectMultiScale(
            grayImg, scaleFactor=1.1, minNeighbors=13)
        return face, grayImg
    except:
        return [], []


def rectangle(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)


def Text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y-25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
