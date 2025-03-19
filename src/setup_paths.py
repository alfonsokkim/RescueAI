import os
import wget
import glob

# Set base directory to 'object_detection'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths relative to `object_detection`
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'images'),
    'IMAGE_TRAIN_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'images', 'train'),
    'IMAGE_TEST_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'images', 'test'),
    'MODEL_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join(BASE_DIR, 'Tensorflow', 'protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join(paths['CHECKPOINT_PATH'], 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Creating directories
for path in paths.values():
    os.makedirs(path, exist_ok=True)
print("Necessary directories created.")

# Download Pretrained Model
pretrained_model_tar = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz')

if not os.path.exists(pretrained_model_tar):
    print("Downloading Pretrained Model...")
    wget.download(PRETRAINED_MODEL_URL, pretrained_model_tar)
    print("\nDownload Complete.")
else:
    print("Pretrained model already downloaded.")

# Extract Pretrained Model
if os.name == 'nt':  # Windows
    print("Extracting model files...")
    os.system(f'tar -zxvf "{pretrained_model_tar}" -C "{paths["PRETRAINED_MODEL_PATH"]}"')
    print("Extraction Complete.")

# Create Label Map File
labels = [
    {'name': 'Healthy', 'id': 1},
    {'name': 'Wounded', 'id': 2}
]

print("Generating label map...")

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write(f"item {{\n  id: {label['id']}\n  name: \"{label['name']}\"\n}}\n\n")

print(f"Label map created at {files['LABELMAP']}")

# Create train and test image directories
os.makedirs(paths['IMAGE_TRAIN_PATH'], exist_ok=True)
os.makedirs(paths['IMAGE_TEST_PATH'], exist_ok=True)
print("Training and testing directories created.")

# Ensure TFRecord script exists
if not os.path.exists(files['TF_RECORD_SCRIPT']):
    print("Downloading generate_tfrecord.py...")
    os.system(f'git clone https://github.com/nicknochnack/GenerateTFRecord "{paths["SCRIPTS_PATH"]}"')
    print("generate_tfrecord.py downloaded.")

# Convert training dataset (Including Subdirectories) to TFRecord
print("Converting training annotations to TFRecord format...")

# Collect all XMLs inside train (including healthy & wounded subfolders)
train_xmls = glob.glob(os.path.join(paths["IMAGE_TRAIN_PATH"], "**", "*.xml"), recursive=True)

# Generate TFRecord command
train_cmd = f'python "{files["TF_RECORD_SCRIPT"]}" -x "{paths["IMAGE_TRAIN_PATH"]}" -l "{files["LABELMAP"]}" -o "{os.path.join(paths["ANNOTATION_PATH"], "train.record")}"'

if len(train_xmls) > 0:
    os.system(train_cmd)
    print("Training TFRecord generated.")
else:
    print("No XML files found in train dataset. Ensure images are labeled.")

# Convert test dataset to TFRecord
print("Converting test annotations to TFRecord format...")

# Collect all XMLs inside test (including healthy & wounded subfolders)
test_xmls = glob.glob(os.path.join(paths["IMAGE_TEST_PATH"], "**", "*.xml"), recursive=True)

# Generate TFRecord command
test_cmd = f'python "{files["TF_RECORD_SCRIPT"]}" -x "{paths["IMAGE_TEST_PATH"]}" -l "{files["LABELMAP"]}" -o "{os.path.join(paths["ANNOTATION_PATH"], "test.record")}"'

if len(test_xmls) > 0:
    os.system(test_cmd)
    print("Test TFRecord generated.")
else:
    print("No XML files found in test dataset. Ensure images are labeled.")

print("TFRecord conversion complete.")
