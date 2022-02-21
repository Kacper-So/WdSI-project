# WdSI-project
PL: Projekt, przedmiot: Wprowadzenie do Sztucznej Inteligencji , kierunek: Automatyka i Robotyka (AiR), semestr: 5, Uwagi: python

Projekt ma na celu realizacje zadania klasyfikacji oraz detekcji.

Klasyfikacja została zrelizowana w następujący sposób:
1. Wczytanie danych train oraz test,
2. Wycięcie odpowiednich fragmentów z poszczególnch obrazów ze zbioru treningowego oraz testowego,
3. Ekstrakcja cech poszczególnych obrazów ze zbioru trenignowego oraz stworzenie słownika,
4. Trenowanie,
5. Odpowiednia procedura przyjmowania danych wejściowych,
6. Klasyfikacja wskazanych w danych wejściowych obrazów z folderu test,
7. Odpowiednia procedura wypisywania danych wyjściowych.

Detekcja została zrealizowana w następujący sposób
1. Wczytanie danych train oraz test,
2. Filtracja obrazu przy pomocy odcięcia niektórych składowych koloru w skali hsv, w celu wykrycia obszarów czerwonych (czerwony okrąg na znaku),
3. Użycie wbudowanego algorytmu opencv MSER (Maximally stable extremal regions) w celu determinacji potencjalnych obszarów występowania znaku.
   Algorytm ten wykrywa obszary obrazu które utrzmują swój kształt podczas progowania (thresholding) obrazu na kilku poziomach,
5. Eskalacja wykrytych obszarów do odpowiednich bounding boxes
4. Wycięcie wykrytych obszarów z obrazu
5. Wycięcie odpowiednich fragmentów z poszczególnch obrazów ze zbioru treningowego,
6. Ekstrakcja cech poszczególnych obrazów ze zbioru trenignowego oraz stworzenie słownika,
7. Trenowanie,
8. Odpowiednia procedura przyjmowania danych wejściowych,
9. Klasyfikacja wcześniej wykrytych i wyciętych obszarów,
10. Odpowiednia procedura wypisywania danych wyjściowych.

Wnioski:
Algorytm klasyfikacji uzyskał dokładność około 93%
Algorytm detekcji znaków jest zawodny. Wynika to prawdopodobnie ze złego przyjęcia metodologii wykrywania obszarów podejrzanych o zawieranie w sobie znaku oraz przyjęcia złych parametrów przyjętej metodologii.
Algorytm zachowuje się inaczej w zależności od przyjętego zakresu koloru odcinanego w filtrze wstępnym oraz intensywności przeszukiwania obrazu (dzielenie obrazu na części).
Na pewno istnieją ustawienia dla których algorytm wykrywa znaki z dużą dokładnością, natomiast ciężkie jest ich określenie metodą prób i błędów.
Program wypisuje również zdecydowanie za dużo znaków w stosunku do prawidłowej wartości. Wynika to z algorytmu MSER, który wykrywa wiele podobnych do siebie miejscem obszarów oraz przesyła je następnie do klasyfikacji.
Rozwiązaniem tego problemu byłoby przebadanie wszystkich kombinacji otrzymanych obszarów zaklasyfikowanych jako znak ograniczenia prędkości pod kątem wskaźnika IoU oraz usunięcie multiplikatów.

Edit:
Niestety nie znam bezpośredniej przyczyny błędu który sprawia brak kompilacji u Pana. Na moim komputerze wszystko działa. Poprawiłem jeszcze kilka rzeczy które mogą powodować ten błąd, natomiast nie znam konkretnej przyczyny.