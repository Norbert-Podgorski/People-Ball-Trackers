# People-Ball-Trackers

<br>

## NOTATKI

### WYNIKI - YOLO (CONFIDENCE THRESHOLD - 0.5):
#### PRETRAINED:
- PASSES_1: 244 klatki, 489 detekcji ludzi (1 false positive - klatka nr 133, pozostałe true positive), 227 detekcji piłek (0 false positive, 17 false negative)
- PASSES_2: 342 klatki, 684 detekcje ludzi (wszystkie true positive), 235 detekcji piłek (0 false positive, 107 false negetive)
- PASSES_3: 298 klatek, 589 detekcji ludzi (7 połączonych detekcji - w dalszej części traktowane jako brak predykcji czyli 14 false negative, pozostałe true positive), 264 detekcje piłek (0 false positive, 34 false negative)
#### TRAINED:
- PASSES_1: 244 klatki, 488 detekcji ludzi (wszystkie true positive), 240 detekcji piłek (0 false positive, 4 false negative)
- PASSES_2: 342 klatki, 684 detekcje ludzi (wszystkie true positive), 265 detekcji piłek (0 false positive, 77 false negetive)
- PASSES_3: 298 klatek, 589 detekcji ludzi (7 połączonych detekcji - w dalszej części traktowane jako brak predykcji czyli 14 false negative, pozostałe true positive), 273 detekcje piłek (0 false positive, 25 false negative)
  
<br><br>

## TO-DO:
- Dodać nowe struktury na detekcje ludzi oraz piłki tak żeby ludzie mieli indeksy i byli śledzeni osobno (musi być kompatybilne z obecną strukturą dla możliwości logowania i wizualizacji), wspólna detekcja dla dwóch osób będzie przypisana tylko do jednej osoby - DaSiam ma to naprawić
- Algorytm do wyliczania maksymalnego przesunięcia w pikselach na podstawie odległości od obiektów oraz frame'u nagrania
- Trackery na podstawie detekcji na Trained YOLO ze zmniejszonym znacznie confidence (mogą być false positive, tracker ma je filtrować)
- DaSiam
- ...
