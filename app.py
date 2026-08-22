import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Mine Detection System", page_icon="🌊", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #eaf6ff;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4fc3f7;
        text-align: center;
        text-shadow: 0 0 30px rgba(79,195,247,0.5);
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #8fc6ff;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        opacity: 0.8;
    }

    .alert-mine {
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,82,82,0.1));
        border: 2px solid rgba(255,107,107,0.4);
        color: #ff6b6b;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 20px 30px;
        border-radius: 16px;
        text-align: center;
        text-shadow: 0 0 10px rgba(255,107,107,0.5);
        margin: 1rem 0;
    }

    .alert-clear {
        background: linear-gradient(135deg, rgba(100,255,218,0.1), rgba(79,195,247,0.1));
        border: 2px solid rgba(100,255,218,0.3);
        color: #64ffda;
        font-size: 1.1rem;
        font-weight: 500;
        padding: 16px 24px;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 24px;
        margin: 1rem 0;
    }

    .card-title {
        font-size: 1.2rem;
        color: #64ffda;
        font-weight: 600;
        margin-bottom: 1rem;
        border-left: 4px solid #64ffda;
        padding-left: 10px;
    }

    .summary-item {
        background: rgba(100,255,218,0.1);
        border: 1px solid rgba(100,255,218,0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .summary-label { color: #8fc6ff; font-size: 0.9rem; }
    .summary-count { color: #64ffda; font-size: 2rem; font-weight: 700; }

    .footer {
        text-align: center;
        color: #8fc6ff;
        opacity: 0.6;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00bcd4 0%, #2196f3 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2196f3 0%, #00bcd4 100%);
        box-shadow: 0 6px 20px rgba(33,150,243,0.5);
    }

    [data-testid="stFileUploader"] {
        background: rgba(79,195,247,0.05);
        border: 2px dashed rgba(79,195,247,0.4);
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), 'best_new.pt')
    return YOLO(path)

model = load_model()

st.markdown('<div class="main-title">🌊 Underwater Mine Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Detection for Marine Safety</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a sonar image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image.thumbnail((416, 416))

    if st.button("🔍 Detect"):
        with st.spinner("Analyzing image..."):
            results = model.predict(source=image, conf=0.25, save=False, imgsz=416)
            r = results[0]
            boxes = r.boxes
            annotated = Image.fromarray(r.plot()[:, :, ::-1])

        st.markdown('<div class="card-title">Detection Result</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(annotated, caption="Annotated", use_container_width=True)

        if boxes is None or len(boxes) == 0:
            st.markdown('<div class="alert-clear">✅ No mines detected.</div>', unsafe_allow_html=True)
        else:
            class_counts = {}
            rows = []
            alert = ""

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = round(float(box.conf[0]) * 100, 2)
                label = model.names.get(cls_id, f"Class {cls_id}")
                rows.append({"Class": label, "Confidence": f"{conf}%"})
                class_counts[label] = class_counts.get(label, 0) + 1
                if label.lower() == "milco" and conf > 60:
                    alert = "⚠️ Mine ahead!"

            if alert:
                st.markdown(f'<div class="alert-mine">{alert}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-clear">🔍 Detection complete.</div>', unsafe_allow_html=True)

            st.markdown('<div class="card-title">Detection Details</div>', unsafe_allow_html=True)
            st.markdown("""
            <style>
            [data-testid="stTable"] table { color: #eaf6ff; width: 100%; }
            [data-testid="stTable"] th { color: #64ffda; background: rgba(100,255,218,0.1); font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase; }
            [data-testid="stTable"] td { color: #eaf6ff; border-bottom: 1px solid rgba(255,255,255,0.1); }
            [data-testid="stTable"] tr:hover td { background: rgba(100,255,218,0.05); }
            </style>
            """, unsafe_allow_html=True)
            st.table(rows)

            st.markdown('<div class="card-title">Detection Summary</div>', unsafe_allow_html=True)
            cols = st.columns(len(class_counts))
            for col, (label, count) in zip(cols, class_counts.items()):
                with col:
                    st.markdown(f'<div class="summary-item"><div class="summary-label">{label}</div><div class="summary-count">{count}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Developed by <strong>Devansh, Shreyans</strong></div>', unsafe_allow_html=True)
