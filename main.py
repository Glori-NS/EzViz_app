from my_speech_recognition import listen  
from text_to_speech import speak 
from object_detection import ObjectDetector
import cv2
import os

def main():
    base_path = r'C:\Users\Gloribeth\AccessibleProductIdentifier'
    model_name = 'ssd_mobilenet_v2_fpnlite_320x320'
    model_path = os.path.join(base_path, 'models', model_name, 'saved_model')
    label_map_path = os.path.join(base_path, 'config', 'mscoco_label_map.pbtxt')

    detector = ObjectDetector(model_path, label_map_path)
    
    # Start video from the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set video width and height for consistency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Start by listening for commands
        command = listen()
        if "start identification" in command.lower():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                continue  # Skip the rest of the loop and listen for commands again

            # Perform object detection
            detected_frame, detected_objects = detector.detect_objects(frame)

            # Provide feedback for detected objects
            for obj in detected_objects:
                speak(f"Detected a {obj}")
            
            # Display the resulting frame
            cv2.imshow('Object Detection', detected_frame)
        
        elif "stop" in command.lower():
            break
        elif "help" in command.lower():
            # Provide help information
            speak("You can ask me to 'start identification' or 'stop' to exit.")

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
