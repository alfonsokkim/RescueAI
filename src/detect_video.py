import cv2
import numpy as np
import tensorflow as tf
import os
from object_detection.utils import visualization_utils as viz_utils
from setup_paths import paths, files
from train_model import detect_fn

# Set video path
video_path = r"F:/videos/Desktop/Desktop 2025.02.27 - 12.21.10.04.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video exists
if not os.path.exists(video_path):
    print("Error: The file does not exist at the given path.")
    exit()

# Check if capture is successful
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set output video path
output_path = r"F:/videos/Desktop/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.8,
        agnostic_mode=False
    )

    out.write(image_np_with_detections)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Output saved to:", output_path)
