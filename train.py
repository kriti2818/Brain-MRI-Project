import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

train_path = "dataset/Training"
test_path = "dataset/Testing"

# -------------------------------
# DATA LOADING
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -------------------------------
# MODEL (Transfer Learning)
# -------------------------------
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------------
# EVALUATE
# -------------------------------
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc:.4f}")

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("brain_tumor_model.keras")

print("\nModel saved as brain_tumor_model.h5")