import streamlit as st
import os
from utils import extract_zip, clear_directory
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("📸 AutoML Vision - Image Classifier Trainer")

# Upload dataset
uploaded_file = st.file_uploader("Upload ZIP of images (class-wise folders)", type="zip")

if uploaded_file is not None:
    clear_directory("data/")
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.read())
    extract_zip("temp.zip", "data/")
    st.success("✅ Dataset uploaded and extracted.")

    # Data generators
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_gen = datagen.flow_from_directory(
        "data/",
        target_size=(224, 224),
        batch_size=32,
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        "data/",
        target_size=(224, 224),
        batch_size=32,
        subset='validation'
    )

    # Model
    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    st.subheader("🚀 Training Model...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=5)

    # Plot
    st.subheader("📈 Accuracy")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Acc')
    ax.plot(history.history['val_accuracy'], label='Val Acc')
    ax.legend()
    st.pyplot(fig)

    # Save
    if st.button("💾 Save Model"):
        model.save("models/automl_model.h5")
        st.success("Model saved to `models/automl_model.h5`.")
