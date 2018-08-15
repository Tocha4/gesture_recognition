# CNN zur Erkennung von Handgesten und Aufnahme von Abfolgen
Mit diesem Projekt habe ich mich in die Bibliothek ```tensorflow``` eingearbeitet. Dieses Repository zeigt meinen den Weg von der Aufnahme der Trainingsdaten, zum Modell bis hin zu der Anwendung.
## Motivation zur Gestenerkennung
Die Grundidee war es, eine Software zu entwickeln, mit der es möglich ist Hardwarekomponenten oder z.B. Präsentations-Software (Jupyter Slides, PowerPoint) zu steuern. Neben einfachen Gesten sollen auch Abfolgen aufgenommen und verarbeitet werden. 

## Vorgehen
Mein Vorgehen lässt sich in drei Arbeitspakete unterteilen. __Zuerst__ habe ich Trainings- und Validierungsdaten gesammelt. Dazu habe ich eine Webcam und [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html) ```cv2``` verwendet. Als __zweites__ habe ich ein *convolutional neural network* mit ```tensorflow``` erstellt und trainiert. Den Lernfortschritt habe ich mit ```tensorboard``` beobachtet. Im __dritten__ Schritt wird das CNN-Modell zur direkten Analyse von Webcam-Bildern verwendet. Für jeden dieser drei Schritte habe ich jeweils ein Jupyter-Notebook mit einer detailliert Beschreibung erstellt.

1. Schritt: [Daten_sammeln](https://github.com/Tocha4/gesture_recognition/blob/master/beschreibung/Daten_sammeln.ipynb)
2. Schritt: [Modell](https://github.com/Tocha4/gesture_recognition/blob/master/beschreibung/Modell.ipynb)
3. Schritt: [Gesten_erkennen](https://github.com/Tocha4/gesture_recognition/blob/master/beschreibung/Gesten_erkennen.ipynb)

[![IMAGE ALT TEXT HERE](movie.gif)](http://www.youtube.com/watch?v=BbtzxUtnSDU)

