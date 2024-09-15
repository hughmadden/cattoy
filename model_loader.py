import cv2

def load_model(model_name="ssd_mobilenet_v2"):
    if model_name == "ssd_mobilenet_v2":
        net = cv2.dnn.readNet('models/ssd_mobilenet_v2_coco.pb', 'models/ssd_mobilenet_v2_coco.pbtxt')
        class_names = [line.strip() for line in open('models/coco.names')]
    # You can add more models here with different file paths if needed.
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return net, class_names

