import streamlit as st
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Сегментация леса (ResNet34 Unet)", layout="wide")
st.title("🌲 Сегментация леса на аэрофотоснимках (Unet + ResNet34)")

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # веса энкодера не нужны, т.к. мы грузим свои веса
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load("best_model-3.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Загрузка изображения пользователем или по ссылке ---
st.header("1. Загрузите изображение для сегментации")
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
        img = Image.open(file).convert("RGB")
        images.append((img, file.name))
elif img_url:
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        images.append((img, "from_url.jpg"))
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")

# --- Сегментация ---
def predict_mask(img, model):
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    image_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred_logits = model(image_tensor)
        pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
    return pred_mask.squeeze().cpu().numpy()

if images:
    st.header("2. Результаты сегментации")
    cols = st.columns(len(images))
    for idx, (img, name) in enumerate(images):
        with cols[idx]:
            st.image(img, caption=f"Исходное: {name}", use_container_width=True)
            mask = predict_mask(img, model)
            st.image(mask, caption="Маска леса", use_container_width=True, clamp=True)
            # Наложение маски на изображение
            img_np = np.array(img.resize((256, 256)))
            overlay = img_np.copy()
            overlay[mask > 0.5] = [0, 255, 0]  # зелёный цвет для леса
            st.image(overlay, caption="Сегментация (наложение)", use_container_width=True)


st.markdown("""
---
**Модель:**  
Unet с энкодером ResNet-34 (предобученные веса ImageNet).  
Файл весов: `best_model-3.pth`.

**Особенности:**  
- Используется другая архитектура по сравнению с классическим U-Net.
- Энкодер ResNet-34 позволяет лучше извлекать признаки из изображений.
- Сегментация проводится по тем же данным, что и на предыдущей странице.
""")