# =================================================================
# --- 1️⃣ BIBLIOTHEKEN IMPORTIEREN ---
# =================================================================
# SSL-Zertifikat-Konfiguration für macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# TensorFlow/Keras: Framework für maschinelles Lernen
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist  # MNIST: Datensatz mit handgeschriebenen Ziffern

# Matplotlib: Bibliothek zur Visualisierung
import matplotlib.pyplot as plt

# =================================================================
# --- 2️⃣ MNIST-DATEN LADEN ---
# =================================================================
# SSL-Zertifikatproblem umgehen für den Download
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Laden der MNIST-Daten:
# x_train: 60.000 Trainingsbilder (28x28 Pixel)
# y_train: Die korrekten Ziffern (Labels) für die Trainingsbilder
# x_test:  10.000 Testbilder
# y_test:  Die korrekten Ziffern für die Testbilder
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Ausgabe der Datensatzgrößen:
# x_train.shape[0] gibt die Anzahl der Trainingsbilder (60.000)
print("Number of training images:", x_train.shape[0])

# x_test.shape[0] gibt die Anzahl der Testbilder (10.000)
print("Number of testing images:", x_test.shape[0])

# x_train.shape[1:] gibt die Bildgröße (28, 28)
# [1:] bedeutet: nimm alle Dimensionen außer der ersten
# Also: shape[1] = Höhe (28), shape[2] = Breite (28)
print("Shape of each image:", x_train.shape[1:])  # (28, 28)

# --- 3️⃣ Beispielbild anzeigen ---
# plt.imshow zeigt ein Bild an
# x_train[20]: Wählt das 20. Bild aus dem Trainingsdatensatz
# cmap='gray': Zeigt das Bild in Graustufen an (da MNIST-Bilder Graustufen sind)
plt.imshow(x_train[20], cmap='gray')

# Zeigt die korrekte Ziffer (Label) über dem Bild an
# y_train[20]: Das Label (richtige Ziffer) für das 20. Bild
# f"Label: {y_train[20]}" erstellt einen String wie "Label: 5"
plt.title(f"Label: {y_train[20]}")

# Zeigt das Bild tatsächlich an
# Ohne show() würde das Bild nicht angezeigt werden
plt.show()

# =================================================================
# --- 4️⃣ NORMALISIERUNG DER DATEN ---
# =================================================================s
# Skalierung der Pixelwerte von 0-255 auf 0-1
# Dies verbessert das Training, da alle Werte im gleichen Bereich liegen
x_train = x_train / 255.0
x_test = x_test / 255.0

# =================================================================
# --- 5️⃣ AUFBAU DES NEURONALEN NETZES ---
# =================================================================
# Sequential: Ein lineares Stapeln von Schichten

model = keras.models.Sequential([
    # Eingabeschicht:
    # Wandelt 28x28 Bilder in einen Vektor mit 784 Werten um
    keras.layers.Flatten(input_shape=(28, 28)),  # 28x28 → 784 Neuronen
    
    # Versteckte Schicht: 
    # 128 Neuronen mit ReLU-Aktivierung
    # ReLU = Rectified Linear Unit: f(x) = max(0, x)
    keras.layers.Dense(128, activation='relu'),
    
    # Ausgabeschicht:
    # 10 Neuronen (für Ziffern 0-9) mit Softmax-Aktivierung
    # Softmax wandelt Werte in Wahrscheinlichkeiten um (Summe = 1)
    keras.layers.Dense(10, activation='softmax')
])

# =================================================================
# --- 6️⃣ MODELL KONFIGURATION ---
# =================================================================
model.compile(
    # Adam: Ein fortgeschrittener Optimierungsalgorithmus
    optimizer='adam',
    # Verlustfunktion: Misst, wie falsch die Vorhersagen sind
    loss='sparse_categorical_crossentropy',
    # Metriken: Was wird während des Trainings gemessen
    metrics=['accuracy']  # Genauigkeit = Anteil der richtigen Vorhersagen
)

# =================================================================
# --- 7️⃣ TRAINING DES MODELLS ---
# =================================================================
print("\nTraining startet...\n")

# Eine Epoche = Ein kompletter Durchlauf durch ALLE Trainingsdaten
# In jeder Epoche:
# 1. Werden alle 60.000 Bilder durchs Netzwerk geschickt
# 2. Für jedes Bild werden die Gewichte leicht angepasst
# 3. Die Genauigkeit (accuracy) sollte steigen
# 4. Der Verlust (loss) sollte sinken
model.fit(x_train, y_train, epochs=5)


# =================================================================
# --- 8️⃣ EVALUIERUNG DES MODELLS ---
# =================================================================
# Testen auf den 10.000 Testbildern, die das Modell noch nie gesehen hat
# Dies zeigt, wie gut das Modell auf neue Daten generalisiert
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTestgenauigkeit: {test_acc:.4f}")

# =================================================================
# --- 9️⃣ VORHERSAGE FÜR EIN EINZELNES BILD ---
# =================================================================
import numpy as np

# Wähle ein Testbild aus
index = 20  # Sie können diesen Index ändern (0-9999)

# Zeige das Bild an
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Richtige Zahl: {y_test[index]}")
plt.show()

# Mache eine Vorhersage
# Das Modell gibt Wahrscheinlichkeiten für jede Ziffer (0-9) zurück
# argmax wählt die Ziffer mit der höchsten Wahrscheinlichkeit
prediction = model.predict(np.array([x_test[index]]))
print("Vorhersage:", np.argmax(prediction))
