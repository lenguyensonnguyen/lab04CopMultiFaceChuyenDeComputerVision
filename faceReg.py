import cv2
from mtcnn.mtcnn import MTCNN
import os

detector = MTCNN()

faceID = 0
for file in os.listdir("images"):
    if file.endswith("jpg"):
        image = cv2.imread("images/" + file)
        image_for_crop = cv2.imread("images/" + file)
        results = detector.detect_faces(image)

        for result in results:
            bounding_box = result['box']
            img_crop = image_for_crop[bounding_box[1]:bounding_box[1] + bounding_box[3],
                       bounding_box[0]:bounding_box[0] + bounding_box[2]]
            cv2.imwrite("faces/face_" + faceID.__str__() + ".jpg", img_crop)
            faceID += 1
