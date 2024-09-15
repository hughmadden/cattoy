import cv2
import cvzone

thres = 0.6
nmsThres = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load COCO class names
classNames = []
classFile = 'models/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

# Load model configuration and weights
configPath = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'models/frozen_inference_graph.pb'

# Load the pre-trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    if len(classIds) > 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Check if the detected class is a person (class ID 1 for COCO dataset)
            if classId == 1:
                # Draw bounding box and label for the person
                cvzone.cornerRect(img, box)
                cv2.putText(img, f'Person {round(conf*100,2)}%',
                            (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
