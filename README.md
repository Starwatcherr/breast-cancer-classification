# Brustkrebs-Klassifizierung

Prüfungsprojekt im Fach Machine Learning, Sommersemester 2026, HSHL (Prof. Dr. Klaus Brinker).

Eigene Implementierung eines linearen Klassifikators (ADALINE/Delta-Regel) mit Gradientenabstieg zur binären Klassifikation des Wisconsin Diagnostic Breast Cancer Datensatzes.

## Inhalt
- `Brustkrebs-Klassifizierung.py`: Hauptcode (Modell, Training, Evaluation, Visualisierungen)
- `breast-cancer_train.csv`, `breast-cancer_test.csv`: Datensatz
- `img/`: Zielordner für generierte Plots (Inhalt wird beim Ausführen erzeugt)

## Ausführung
```
python Brustkrebs-Klassifizierung.py
```
Die generierten Plots werden im `img/` Ordner gespeichert. Endergebnis: Testgenauigkeit 95,2%.