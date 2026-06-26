def fehler(w0, w1, x0, x1, y):
    # lineare Ausgabe: gewichtete Summe
    vorhersage = w0 * x0 + w1 * x1
    # halber quadratischer Fehler
    return 0.5 * (y - vorhersage) ** 2

# ein einzelnes Trainingsbeispiel
x0, x1 = 2.0, 3.0
y = 1.0

# aktuelle Gewichte
w0, w1 = 0.5, 0.5

# wie empfindlich ist der Fehler auf w0?
# wir drehen w0 um ein winziges bisschen hoch und schauen, wie sich der Fehler ändert
winzig = 0.0001

fehler_jetzt   = fehler(w0,         w1, x0, x1, y)
fehler_gedreht = fehler(w0 + winzig, w1, x0, x1, y)

steigung = (fehler_gedreht - fehler_jetzt) / winzig
print(steigung)