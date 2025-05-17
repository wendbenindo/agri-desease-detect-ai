import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from preprocess import load_data

# Paramètres
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'plant_disease_model.h5'))

# Chargement des données
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
train_data, val_data = load_data(data_dir)

# Création du modèle basé sur MobileNetV2
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # On ne réentraîne pas les couches de base

# Ajout de couches personnalisées
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilation du modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Sauvegarde
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\n Modèle sauvegardé sous : {MODEL_PATH}")
