from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Folder setup
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLO model
MODEL_PATH = 'best.pt'

try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # --- Run YOLO Prediction ---
    result_name = f"result_{os.path.splitext(filename)[0]}"
    result_dir = os.path.join(app.config['RESULT_FOLDER'], result_name)
    print(f"YOLO saving results to directory: {result_dir}") # Debug print

    results = model.predict(
        source=filepath,
        conf=0.25,
        save=True,
        project=app.config['RESULT_FOLDER'], # Base project directory
        name=result_name # Subdirectory name for results
    )

    # --- Process Results ---
    predictions = []
    class_counts = {}
    alert_message = ""
    # Default to original image path
    image_to_display_path = filepath

    if len(results) > 0:
        r = results[0]
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:
            # --- Objects Detected ---
            print(f"Found {len(boxes)} detections.") # Debug print
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                label = model.names.get(cls_id, f"Class {cls_id}")

                predictions.append({'label': label, 'confidence': round(conf, 2)})
                class_counts[label] = class_counts.get(label, 0) + 1

                if label.lower() == "milco" and conf > 60:
                    alert_message = "⚠️ Mine ahead!"

            # --- Find the Annotated Image ---
            # YOLO saves the annotated image in the specified directory (result_dir).
            # We need to find it there.
            print(f"Searching for annotated image in: {result_dir}") # Debug print
            found_annotated_image = False

            if os.path.exists(result_dir): # Check if the directory exists first
                print(f"Contents of {result_dir}: {os.listdir(result_dir)}") # Debug print
                for file_in_dir in os.listdir(result_dir):
                    if file_in_dir.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        # Found the annotated image
                        image_to_display_path = os.path.join(result_dir, file_in_dir)
                        print(f"Found annotated image: {image_to_display_path}") # Debug print
                        found_annotated_image = True
                        break # Exit loop after finding the first image
            else:
                 print(f"Warning: Expected result directory does not exist: {result_dir}") # Debug print


            if not found_annotated_image:
                 print(f"Warning: Annotated image not found in {result_dir}. Using original image.")
                 # If for some reason the annotated image isn't found, fall back to original
                 alert_message = "Prediction completed, but annotated image not found. Showing original."

        else:
            # --- No Objects Detected by Model ---
            print("No detections found by the model.") # Debug print
            alert_message = "No objects detected."
            # Keep image_to_display_path as the original filepath

    else:
        # --- Prediction Object Error (unlikely but handled) ---
        print("Warning: YOLO results object was empty or unexpected.")
        alert_message = "Prediction failed."
        # Keep image_to_display_path as the original filepath

    # --- Prepare Image Path for HTML (Final Fix) ---
    # Flask serves static files from the 'static' folder.
    # The image_to_display_path is currently an absolute path.
    # We need to convert it to a path relative to the 'static' folder for the HTML img src.
    # Since we know the structure, we can construct the path manually.
    try:
        # Get the absolute path of the image to display
        abs_image_to_display_path = os.path.abspath(image_to_display_path)

        # Get the absolute path of the static folder
        abs_static_folder_path = os.path.abspath(app.static_folder)

        # Calculate the relative path from the static folder to the image file
        # This should give us a path like 'uploads/0001_2015_aug_p1_c0539.jpg' or 'results/result_0001_2015_aug_p1_c0539/0001_2015_aug_p1_c0539.jpg'
        rel_path_from_static = os.path.relpath(abs_image_to_display_path, start=abs_static_folder_path)

        # The path for the HTML img src tag should be relative to the 'static' folder.
        # Flask will serve it from http://127.0.0.1:5000/static/rel_path_from_static
        img_path_for_html = rel_path_from_static.replace("\\", "/")

        print(f"Calculated image path for HTML (relative to static): {img_path_for_html}") # Debug print

    except Exception as e:
        print(f"Error creating image path for HTML: {e}")
        # Fallback: Use the original image path relative to the static folder
        # This is a last resort if the above fails
        try:
            # Try to get the path relative to static for the original image
            abs_original_filepath = os.path.abspath(filepath)
            rel_original_path = os.path.relpath(abs_original_filepath, start=abs_static_folder_path)
            img_path_for_html = rel_original_path.replace("\\", "/")
            print(f"Using original image path as fallback: {img_path_for_html}") # Debug print
        except Exception as e2:
            # Final fallback
            print(f"Final fallback: Using 'uploads/default_image.jpg'")
            img_path_for_html = "uploads/default_image.jpg"

    # --- Render Template ---
    return render_template(
        'result.html',
        predictions=predictions,
        alert=alert_message,
        img_path=img_path_for_html, # Send the correctly formatted path to the template
        class_counts=class_counts
    )

if __name__ == '__main__':
    app.run(debug=True)
