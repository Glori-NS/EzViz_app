import cv2
import tensorflow as tf
import numpy as np

def load_label_map(label_map_path):
    labels = {}
    with open(label_map_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "id:" in line:
                class_id = int(line.split('id:')[1].strip())
            if "display_name:" in line:
                class_name = line.split('display_name:')[1].strip().strip('"')
                labels[class_id] = class_name
    return labels

class ObjectDetector:
    def __init__(self, model_path, label_map_path):
        self.model = tf.saved_model.load(model_path)
        self.labels = load_label_map(label_map_path)

    def preprocess_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor

    def draw_bounding_box(self, image, box, class_id, score):
        y_min, x_min, y_max, x_max = box
        height, width, _ = image.shape
        y_min, x_min, y_max, x_max = int(y_min * height), int(x_min * width), int(y_max * height), int(x_max * width)
        
        label = self.labels.get(class_id, f"Class ID {class_id}")
        label_text = f"{label}: {round(score * 100, 2)}%"
        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y_min = max(y_min - label_size[1], 0)

        cv2.rectangle(image, (x_min, label_y_min - base_line), (x_min + label_size[0], label_y_min + base_line), (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label_text, (x_min, label_y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


    def postprocess_output(self, image, output_dict, threshold=0.5):
        boxes = output_dict['detection_boxes'][0].numpy()
        scores = output_dict['detection_scores'][0].numpy()
        classes = output_dict['detection_classes'][0].numpy().astype(np.int64)
        
        detected_classes = []
        for i in range(min(20, boxes.shape[0])):  
            if scores[i] > threshold:
                class_id = classes[i]
                class_name = self.labels.get(class_id, 'N/A')
                if class_name not in detected_classes:
                    detected_classes.append(class_name)
                    box = boxes[i]
                    self.draw_bounding_box(image, box, class_id, scores[i])
        return detected_classes

    def detect_objects(self, image):
        preprocessed_image = self.preprocess_image(image)
        output_dict = self.model(preprocessed_image)
        detected_classes = self.postprocess_output(image, output_dict)
        return image, detected_classes  
