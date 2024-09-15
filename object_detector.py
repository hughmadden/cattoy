import cv2

def detect_objects(frame, net, class_names, target_classes, confidence_threshold=0.5):
    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (300, 300), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    height, width = frame.shape[:2]
    objects_detected = []

    for detection in detections[0, 0]:
        confidence = detection[2]
        class_id = int(detection[1])
        if confidence > confidence_threshold:
            label = class_names[class_id]
            if label in target_classes:
                # Get bounding box coordinates
                x1 = int(detection[3] * width)
                y1 = int(detection[4] * height)
                x2 = int(detection[5] * width)
                y2 = int(detection[6] * height)


                # Append detected object info
                objects_detected.append({
                    'label': label,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
    
    return objects_detected
