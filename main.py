import app.src.unautil.utils as ut
import app.src.core.image_homing_guidance as hg
import cv2 as cv
import imutils
import os

if __name__ == '__main__':
    # Load our input image and grab its spatial dimensions
    cap = cv.VideoCapture(0)
    print(os.getcwd())

    # Load labels
    labels_path = 'app/datasets/models/coco.names'
    labels_stream = open(labels_path).read().strip().split("\n")

    config_path = 'app/datasets/models/yolov3/yolov3-tiny.cfg'
    weights_path = 'app/datasets/models/yolov3/yolov3-tiny.weights'

    det = hg.Detector(config_path, weights_path, labels_stream)
    # ui = HomingUI()

    while True:

        # Capture frame by frame
        _, frame = cap.read()
        # Perform image processing

        # Perform object detection
        frame = det.detect(frame)

        # Display
        cv.imshow('Camera', frame)
        # ui.mouse_select()

        # Wait for ESC
        if cv.waitKey(1) == 27:
            break

    # Print source resolution
    print(det.image.shape)

    cap.release()
    cv.destroyAllWindows()
