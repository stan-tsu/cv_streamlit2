import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Детекция ветрогенераторов", layout="wide")

st.title("🎯 Детекция ветрогенераторов (YOLOv11s)")

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    return YOLO("yolo11s_WTD.pt")  # путь к вашей модели

model = load_model()

# --- Загрузка изображений пользователем ---
st.header("1. Загрузите изображения для детекции")
uploaded_files = st.file_uploader(
    "Выберите изображения (можно несколько)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# --- Загрузка по ссылке ---
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

# --- Детекция ---
if images:
    st.header("2. Результаты детекции")
    cols = st.columns(len(images))
    for idx, (img, name) in enumerate(images):
        with cols[idx]:
            st.image(img, caption=f"Исходное: {name}", use_column_width=True)
            results = model.predict(img)
            res_img = results[0].plot()  # визуализация bbox
            st.image(res_img, caption="Детекция", use_column_width=True)

# --- Информация о модели и метриках ---
st.header("3. Информация о модели и метриках")

# Выводим картинку с метриками, если она есть
metrics_img_path = "WTD_graphs.png"
if os.path.exists(metrics_img_path):
    st.image(metrics_img_path, caption="Графики метрик обучения и валидации", use_column_width=True)
else:
    st.info("Файл с графиками метрик (WTD_graphs.png) не найден рядом с main.py.")

st.markdown("""
---
**Процесс обучения:**  
YOLOv11s дообучалась на кастомном датасете ветрогенераторов.  
Использовались стандартные параметры Ultralytics YOLO, оптимизация по mAP, валидация на отдельной выборке.
""")