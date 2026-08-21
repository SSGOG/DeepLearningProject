import gradio as gr
import os
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_new.onnx')
model = YOLO(MODEL_PATH, task='detect')

def predict(image):
    if image is None:
        return None, "No image provided.", ""

    # Resize to reduce memory usage on free tier
    image.thumbnail((416, 416))

    results = model.predict(source=image, conf=0.25, save=False, imgsz=416)
    r = results[0]
    boxes = r.boxes

    annotated = Image.fromarray(r.plot()[:, :, ::-1])

    if boxes is None or len(boxes) == 0:
        return annotated, "No objects detected.", ""

    predictions_md = "| Class | Confidence |\n|-------|------------|\n"
    class_counts = {}
    alert = ""

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = round(float(box.conf[0]) * 100, 2)
        label = model.names.get(cls_id, f"Class {cls_id}")
        predictions_md += f"| **{label}** | {conf}% |\n"
        class_counts[label] = class_counts.get(label, 0) + 1
        if label.lower() == "milco" and conf > 60:
            alert = "⚠️ Mine ahead!"

    summary = "  ".join([f"**{label}**: {count}" for label, count in class_counts.items()])
    details = f"{predictions_md}\n**Summary:** {summary}"

    return annotated, alert or "Detection complete.", details

with gr.Blocks(title="Underwater Mine Detection") as demo:
    gr.Markdown("# 🌊 Underwater Mine Detection System\nAI-Powered Detection for Marine Safety")

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Image")
        out_img = gr.Image(type="pil", label="Detection Result")

    with gr.Row():
        alert_box = gr.Textbox(label="Alert", interactive=False)

    details_box = gr.Markdown()

    btn = gr.Button("Detect", variant="primary")
    btn.click(fn=predict, inputs=inp, outputs=[out_img, alert_box, details_box])

    gr.Markdown("Developed by **Devansh, Shreyans**")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)
