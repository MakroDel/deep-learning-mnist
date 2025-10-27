# MNIST Handschriftenerkennung mit TensorFlow/Keras

Dieses Projekt implementiert ein neuronales Netzwerk zur Erkennung handgeschriebener Ziffern mit dem MNIST-Datensatz.

## 📊 Der MNIST-Datensatz

### Grundstruktur
- **Trainingsdaten**: 60.000 Bilder von handgeschriebenen Ziffern
- **Testdaten**: 10.000 Bilder
- **Bildgröße**: 28×28 Pixel (Graustufen)
- **Zielwerte**: Ziffern von 0 bis 9

### Datenformat im Detail
- **x_train.shape = (60000, 28, 28)**
  - Erste Dimension (60000): Anzahl der Trainingsbilder
  - Zweite Dimension (28): Bildhöhe in Pixeln
  - Dritte Dimension (28): Bildbreite in Pixeln

- **x_test.shape = (10000, 28, 28)**
  - Gleiche Struktur wie x_train, aber mit 10.000 Bildern

### Zugriff auf die Daten
```python
# Ein einzelnes Bild auswählen (Beispiel: das 20. Bild)
bild = x_train[20]        # Gibt eine 28x28 Matrix zurück
label = y_train[20]       # Gibt die entsprechende Ziffer zurück

# Bildanzeige mit Matplotlib
plt.imshow(bild, cmap='gray')     # Zeigt das Bild in Graustufen
plt.title(f"Label: {label}")      # Zeigt die richtige Ziffer als Titel
plt.show()                        # Macht das Bild sichtbar
```

### Pixelwerte
- **Ursprüngliche Werte**: 0 (schwarz) bis 255 (weiß)
- **Nach Normalisierung**: 0.0 (schwarz) bis 1.0 (weiß)
- **Graustufen**: Alle Werte dazwischen sind Grauabstufungen

## 🧠 Architektur des Neuronalen Netzes

Das Netzwerk besteht aus drei Schichten:

1. **Eingabeschicht (Input Layer)**
   - 784 Neuronen (28×28 Pixel)
   - Jedes Neuron repräsentiert einen Pixelwert (0-1)

2. **Versteckte Schicht (Hidden Layer)**
   - 128 Neuronen
   - ReLU-Aktivierungsfunktion: f(x) = max(0, x)
   - Lernt komplexe Muster in den Daten

3. **Ausgabeschicht (Output Layer)**
   - 10 Neuronen (für Ziffern 0-9)
   - Softmax-Aktivierung (gibt Wahrscheinlichkeiten)
   - Summe aller Ausgaben = 1 (100%)

## 🔄 Trainingsprozess

### Was ist eine Epoche?
Eine Epoche ist ein vollständiger Durchlauf durch alle Trainingsdaten. In jeder Epoche:
- Werden alle 60.000 Bilder einmal verarbeitet
- Jedes Bild durchläuft alle Schichten (784 → 128 → 10)
- Die Gewichte werden nach jedem Bild angepasst
- Die Genauigkeit (accuracy) sollte steigen
- Der Verlust (loss) sollte sinken

### Batch-Verarbeitung
- Die 60.000 Bilder werden in Gruppen (Batches) von 32 Bildern verarbeitet
- Daraus ergeben sich 1.875 Batches pro Epoche (60.000 ÷ 32 = 1.875)
- Dies ermöglicht effizienteres Training

## 📈 Metriken und Bewertung

### Accuracy (Genauigkeit)
- Prozentsatz der richtig erkannten Ziffern
- Steigt während des Trainings
- Testgenauigkeit zeigt, wie gut das Modell generalisiert

### Loss (Verlust)
- Maß für die Fehler des Modells
- Sollte während des Trainings sinken
- Hoher Loss = große Fehler
- Niedriger Loss = kleine Fehler

## 🛠 Setup und Ausführung

### 1. Python-Umgebung erstellen
```bash
python3.11 -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

### Warum eine virtuelle Umgebung?
- Isoliert Projektabhängigkeiten
- Verhindert Konflikte zwischen verschiedenen Projekten
- Macht das Projekt einfach reproduzierbar
- Ermöglicht die Verwendung spezifischer Python-Versionen
- Einfaches Aufräumen durch Löschen des venv-Ordners

2. **Abhängigkeiten installieren**:
   ```bash
   pip install tensorflow matplotlib
   ```

3. **Skript ausführen**:
   ```bash
   python mnist.py
   ```

## 🔍 Erweiterte Konzepte

### Datenverarbeitung

#### Normalisierung
- **Ursprüngliche Pixelwerte**: 0-255
  - 0 = Schwarz
  - 255 = Weiß
  - Werte dazwischen = Graustufen

- **Normalisierte Werte**: 0-1
  - Berechnung: `pixel_wert / 255.0`
  - 0.0 = Schwarz
  - 1.0 = Weiß
  - z.B.: 128 → 0.5 (mittleres Grau)

#### Warum normalisieren?
- Neuronale Netze arbeiten besser mit kleinen Zahlen
- Verbessert die numerische Stabilität
- Beschleunigt das Training
- Verhindert, dass große Zahlenwerte das Training dominieren

### Optimierer (Adam)
- Passt die Lernrate automatisch an
- Beschleunigt den Trainingsprozess
- Findet bessere Lösungen

### Aktivierungsfunktionen
- **ReLU**: Verhindert das "Verschwinden" von Gradienten
- **Softmax**: Wandelt Ausgaben in Wahrscheinlichkeiten um

## 📊 Typische Ergebnisse

- **Trainingsgenauigkeit**: ~98-99%
- **Testgenauigkeit**: ~97-98%
- Diese Werte zeigen, dass das Modell gut generalisiert

## 🤔 Häufige Fragen

### Warum 5 Epochen?
- Guter Kompromiss zwischen Trainingszeit und Genauigkeit
- Mehr Epochen bringen oft nur kleine Verbesserungen
- Risiko von Overfitting steigt mit mehr Epochen

### Datenvisualisierung

#### Bilder anzeigen
```python
# Beliebiges Bild aus dem Datensatz anzeigen
index = 42  # Kann jede Zahl von 0 bis 59999 sein
plt.imshow(x_train[index], cmap='gray')
plt.title(f"Dies ist eine {y_train[index]}")
plt.show()
```

#### Nützliche Matplotlib-Funktionen
- `plt.imshow()`: Zeigt das Bild an
- `cmap='gray'`: Verwendet Graustufen
- `plt.title()`: Fügt einen Titel hinzu
- `plt.show()`: Macht das Bild sichtbar

### Was ist Overfitting?
- Das Modell lernt die Trainingsdaten "auswendig"
- Erkennt man an:
  - Hohe Trainingsgenauigkeit
  - Niedrigere Testgenauigkeit
  - Steigender Trainings-Loss

### Warum 128 Neuronen in der versteckten Schicht?
- Experimentell ermittelter guter Wert
- Genug Kapazität für die Mustererkennung
- Nicht zu viele Parameter für schnelles Training