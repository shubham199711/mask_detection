import cv2
import faceRecognization_OpenCV as fcv
import init_model as im
import sys


def classify_face(test_img, x, y, w, h):
    crop_img = test_img[y:y+h, x:x+w]
    crop_img = im.fix_image(crop_img)
    result = model(crop_img)
    label = im.get_label(result)
    return label


def load_model():
    global model
    model = im.get_loaded_model()


if __name__ == "__main__":
    load_model()
    img_location = str(sys.argv[1])
    test_img = cv2.imread(img_location)
    test_img = cv2.copyMakeBorder(
        test_img,
        100,
        100,
        100,
        100,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    face_detect, grayImg = fcv.face_recognization(test_img)
    crop_img = []
    for (x, y, w, h) in face_detect:
        label = classify_face(test_img, x, y, w, h)
        cv2.rectangle(test_img, (x-20, y-20), (x+w+20, y+h+20),
                      (255, 255, 255), thickness=2)
        fcv.Text(test_img, label, x, y)
    resize = cv2.resize(test_img, (700, 700))
    cv2.imshow("Press 0 to Quit", resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
