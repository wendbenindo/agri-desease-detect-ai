import tensorflow as tf

# Charger le modèle Keras
model = tf.keras.models.load_model('../model/plant_disease_model.h5')

# Convertir en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder
with open("../model/plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modèle converti et enregistré en plant_disease_model.tflite")
