import os
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import cv2

# check label extension

image_dir = "C:/Users/erenh/Desktop/task/dataset/train/images"

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        try:
            img = Image.open(os.path.join(image_dir, filename))
            img.verify()
        except Exception as e:
            print(f"Corrupt image file: {filename} -> {e}")

# Check Label Matching:

image_dir = "C:/Users/erenh/Desktop/task/dataset/train/images"
label_dir = "C:/Users/erenh/Desktop/task/dataset/train/labels"

images = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

unmatched_images = images - labels
unmatched_labels = labels - images

if unmatched_images:
    print(f"Images without labels: {unmatched_images}")
if unmatched_labels:
    print(f"Labels without images: {unmatched_labels}")

# Validate Label Content:

for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid format in {label_file} at line {line_num}: {line}")
                else:
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
                            print(f"Out of bounds values in {label_file} at line {line_num}: {line}")
                    except ValueError:
                        print(f"Non-numeric value in {label_file} at line {line_num}: {line}")


#####

# YOLO model
model = YOLO("yolo11n.pt")

# hyperparameter tuning
results = model.tune(data="C:/Users/erenh/Desktop/task/dataset/config.yaml", epochs=25,iterations=10)

# Train the model using your custom dataset
model.train(
    data="C:/Users/erenh/Desktop/task/dataset/config.yaml",
    epochs=100,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.00049,
    warmup_epochs=2.85691,
    warmup_momentum=0.78417,
    box=8.13339,
    cls=0.49902,
    dfl=1.5,
    hsv_h=0.01536,
    hsv_s=0.70627,
    hsv_v=0.39681,
    degrees=0.0,
    translate=0.10596,
    scale=0.48374,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.50964,
    bgr=0.0,
    mosaic=0.97445,
    mixup=0.0,
    copy_paste=0.0
)


metrics = model.val()

# Save
model_path = "C:/Users/erenh/Desktop/task/erenHatipoglu_bestModel.pt"
model.save(model_path) 

# testing and saving predictions
test_dir = Path("C:/Users/erenh/Desktop/task/dataset/test/images")
output_dir = Path("C:/Users/erenh/Desktop/task/dataset/predictions")
output_dir.mkdir(parents=True, exist_ok=True) 

# predictions on test images
for image_path in test_dir.glob("*.*"):  # Matches all file extensions
    results = model.predict(source=str(image_path), save=False)  # Perform inference
    for result in results:
        annotated_image = result.plot()  # Annotate the image with predictions
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated_image)  # Save the annotated image
