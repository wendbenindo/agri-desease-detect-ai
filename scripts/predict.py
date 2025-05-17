import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Charger le modèle
model = load_model('../model/plant_disease_model.h5')

# Dictionnaire des classes (ajuste selon tes classes réelles)
class_names = ['Healthy', 'Maize Leaf Spot', 'Maize Streak', 'Mil Sorgho', 'Sorghum Blight', 'Sorghum Rust']

# Chemin de l’image à tester
img_path = '../test/fig-34a-loose-smut-of-sorghum-b-kernel-smut-of-sorghum-from-bui-511-from-the-division-of-agronomy-college-of-agriculture-davis-suscep-tible-varieties-should-not-be-planted-on-soil-where-root-rot-is-present-rusta-.jpg'

# Charger et prétraiter l’image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # normalisation
img_array = np.expand_dims(img_array, axis=0)

# Prédiction
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Image testée : {os.path.basename(img_path)}")
print("Classe prédite :", predicted_class)
