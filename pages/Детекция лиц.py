import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Детекция лиц и маскировка", layout="wide")

st.title("🧑‍💻 Детекция лиц с маскировкой (YOLO)")

@st.cache_resource
def load_model():
    return YOLO("faces.pt")

model = load_model()

st.header("1. Загрузите изображения для детекции лиц")
uploaded_files = st.file_uploader(
    "Выберите изображения (можно несколько)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

st.subheader("или укажите прямую ссылку на изображение")
img_url = st.text_input("URL изображения")

images = []
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        images.append((img, file.name))
elif img_url:
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        images.append((img, "from_url.jpg"))
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")

def mask_faces(img, results):
    img_masked = img.copy()
    draw = ImageDraw.Draw(img_masked)
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], fill="gray")
    return img_masked


if images:
    st.header("2. Результаты детекции и маскировки")
    cols = st.columns(len(images))
    for idx, (img, name) in enumerate(images):
        with cols[idx]:
            st.image(img, caption=f"Исходное: {name}", use_container_width=True)
            results = model.predict(img)
            res_img = results[0].plot()
            st.image(res_img, caption="Детекция лиц", use_container_width=True)
            masked_img = mask_faces(img, results)
            st.image(masked_img, caption="Маскировка лиц", use_container_width=True)

    # --- Метрики ---
    st.header("3. Информация о модели и метриках")
    metrics_img_path = "faces_graphs.png"
    if os.path.exists(metrics_img_path):
        st.image(metrics_img_path, caption="Графики метрик обучения и валидации", use_container_width=True)
    else:
        st.info("Файл с графиками метрик (faces_graphs.png) не найден рядом с этим файлом.")


st.markdown("""
---
**Процесс обучения:**  
YOLO дообучалась на датасете лиц для задачи детекции.  
После детекции лица автоматически маскируются (замазываются) на изображении.
""")