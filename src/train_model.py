import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from setup_paths import paths, files

# Load pipeline config
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:  # Fix Here
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Set up model parameters
pipeline_config.model.ssd.num_classes = 2  # 2 classes: Healthy & Wounded
pipeline_config.train_config.batch_size = 4  # Adjust based on GPU memory
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    paths['PRETRAINED_MODEL_PATH'], 
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 
    'checkpoint', 
    'ckpt-0'
)
if hasattr(pipeline_config.train_config, "fine_tune_checkpoint_version"):
    pipeline_config.train_config.ClearField("fine_tune_checkpoint_version")

pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(paths['ANNOTATION_PATH'], 'train.record')
]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(paths['ANNOTATION_PATH'], 'test.record')
]

# Save updated config
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:  # Fix Here
    f.write(config_text)

print("Pipeline config updated.")

# Check if TFRecords exist before training
if not os.path.exists(os.path.join(paths['ANNOTATION_PATH'], 'train.record')):
    print("ERROR: train.record file not found. Run setup_paths.py first!")
    exit(1)

if not os.path.exists(os.path.join(paths['ANNOTATION_PATH'], 'test.record')):
    print("ERROR: test.record file not found. Run setup_paths.py first!")
    exit(1)

print("TFRecord files found. Starting training...")

# Train model using TensorFlow Object Detection API
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
TRAIN_DIR = os.path.join(paths['CHECKPOINT_PATH'])

train_cmd = f"""
python "{TRAINING_SCRIPT}" --model_dir="{TRAIN_DIR}" --pipeline_config_path="{files['PIPELINE_CONFIG']}" --num_train_steps=20000 --sample_1_of_n_eval_examples=1 --alsologtostderr
"""

print("Starting training...")
os.system(train_cmd)
