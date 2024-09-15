from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 nano model (nano version for lightweight performance)
model = YOLO('yolov8n.pt')  # You can use yolov8n.pt (Nano) or yolov8s.pt (Small) depending on your performance needs

# Define the COCO class names (YOLOv8 uses the COCO dataset by default)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"
]

# Function to capture and detect cat in webcam feed
def detect_cat_in_webcam():
    # Open webcam (0 means the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv8 model on the frame to perform inference
        results = model(frame)

        # Filter the detections for "cat" (class ID for "cat" is 15 in COCO)
        for result in results:
            for detection in result.boxes:
                class_id = int(detection.cls)
                if COCO_CLASSES[class_id] == 'person':
                    # Extract bounding box information
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])

                    # Draw a rectangle around the detected cat
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{COCO_CLASSES[class_id]} ({detection.conf.item():.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Cat Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Run the cat detection function
if __name__ == "__main__":
    detect_cat_in_webcam()
