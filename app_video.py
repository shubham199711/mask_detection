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
    video_location = str(sys.argv[1])
    if(video_location):

        cap = cv2.VideoCapture(video_location)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('output.mp4',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
        while(cap.isOpened()):
            ret, test_img = cap.read()
            test_img = cv2.copyMakeBorder(
                test_img,
                100,
                100,
                100,
                100,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
            faces_detected, gray_Img = fcv.face_recognization(test_img)
            if(len(faces_detected)):
                for (x, y, w, h) in faces_detected:
                    label = classify_face(test_img, x, y, w, h)
                    cv2.rectangle(test_img, (x-20, y-20), (x+w, y+h),
                                  (255, 255, 255), thickness=2)
                    fcv.Text(test_img, label, x, y)
                resize_img = cv2.resize(test_img, (500, 500))
                result.write(resize_img)
                cv2.imshow("Live Video", resize_img)
                if cv2.waitKey(10) == ord('q'):
                    break
        cap.release()
        result.release()
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        while True:
            ret, test_img = cap.read()
            test_img = cv2.copyMakeBorder(
                test_img,
                100,
                100,
                100,
                100,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
            faces_detected, gray_Img = fcv.face_recognization(test_img)
            if(len(faces_detected)):
                for (x, y, w, h) in faces_detected:
                    label = classify_face(test_img, x, y, w, h)
                    cv2.rectangle(test_img, (x-20, y-20), (x+w, y+h),
                                  (255, 255, 255), thickness=2)
                    fcv.Text(test_img, label, x, y)
                resize_img = cv2.resize(test_img, (500, 500))
                cv2.imshow("Live Video", resize_img)
                if cv2.waitKey(10) == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
