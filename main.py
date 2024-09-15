import cv2
from model_loader import load_model
from object_detector import detect_objects

def process_webcam(model_name, target_classes, save_image=False, output_image="output.jpg"):
    # Load model
    net, class_names = load_model(model_name)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect objects in frame
        detected_objects = detect_objects(frame, net, class_names, target_classes)

        # Draw bounding boxes for detected objects
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            label = f"{obj['label']} ({obj['confidence']:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Optionally save the frame as an image file
        if save_image:
            cv2.imwrite(output_image, frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    target_classes = ["bird", "cat"]  # Specify which objects to detect
    process_webcam(model_name="ssd_mobilenet_v2", target_classes=target_classes, save_image=True)
