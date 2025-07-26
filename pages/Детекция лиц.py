import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os


st.set_page_config(page_title="Детекция лиц (демо)", layout="wide")
st.title("🧑‍💻 Просмотр изображений (детекция отключена)")


st.header("1. Загрузите изображения для просмотра")
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


# --- Просмотр изображений ---
if images:
    st.header("2. Загруженные изображения")
    cols = st.columns(len(images))
    for idx, (img, name) in enumerate(images):
        with cols[idx]:
            st.image(img, caption=f"Изображение: {name}", use_container_width=True)


st.header("3. Информация")
st.markdown("""
---
**Демо-версия:**
Детекция и маскировка лиц отключены из-за ограничений окружения. Здесь можно только просматривать загруженные изображения.
""")