import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"
model_output_path = "trained_model/plant_disease_prediction_model.h5"
class_indices_path = "class_indices.json"

# Create output folder if it doesn't exist
os.makedirs("dataset/trained_model", exist_ok=True)

# Image size & batch
img_height, img_width = 224, 224
batch_size = 32

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation="softmax")
])

# Compile model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_generator,
          validation_data=val_generator,
          epochs=10)

# Save model
model.save(model_output_path)
print(f"✅ Model saved to {model_output_path}")

# Save class indices
with open(class_indices_path, "w") as f:
    json.dump(train_generator.class_indices, f)
print(f"✅ Class indices saved to {class_indices_path}")
