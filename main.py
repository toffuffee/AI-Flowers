import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Загружаем модель MobileNetV2, предварительно обученную на наборе данных ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Функция для предобработки изображения
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img)
    if img.shape[2] == 4:  # если изображение с альфа-каналом, удаляем его
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Функция для предсказания класса изображения
def predict_image_class(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
    return decoded_preds

# Функция для загрузки изображения
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        
        predictions = predict_image_class(file_path)
        result_text = f"{predictions[0][1]} ({predictions[0][2] * 100:.2f}%)"
        result_label.config(text=result_text)

# Создаем основное окно
root = tk.Tk()
root.title("Распознавание растений")

# Добавляем виджеты
image_label = tk.Label(root)
image_label.pack()

upload_button = tk.Button(root, text="Загрузить изображение", command=load_image)
upload_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Запускаем главный цикл обработки событий
root.mainloop()
