import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration des paramètres de prétraitement
IMG_SIZE = (224, 224)  # Taille standard pour MobileNet
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  # 20% pour validation

# referer vers le dossier data contenant en fait le dataset 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

def load_data(data_dir):
    """
    Charge les images à partir d’un dossier contenant des sous-dossiers pour chaque classe.
    Applique un redimensionnement, une normalisation et une séparation entraînement/validation.
    
    :param data_dir: Chemin vers le dossier contenant les dossiers de classes (ex: "data/")
    :return: tuples (train_data, val_data)
    """

    # Création d’un générateur d’images avec normalisation et séparation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    # Chargement des données d'entraînement
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    # Chargement des données de validation
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_data, val_data


if __name__ == "__main__":
    # Point d’entrée si le script est exécuté directement
    data_dir = DATA_DIR   # À adapter si ton dataset est dans un sous-dossier différent
    train_data, val_data = load_data(data_dir)

    # Affichage des informations utiles
    print("\n--- Résumé du dataset ---")
    print(f"Nombre de classes : {len(train_data.class_indices)}")
    print("Classes :", train_data.class_indices)
    print(f"Nombre d'images d'entraînement : {train_data.samples}")
    print(f"Nombre d'images de validation : {val_data.samples}")
