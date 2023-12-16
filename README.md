# People-Ball-Trackers

<br>

## NOTATKI

### WYNIKI - YOLO (CONFIDENCE THRESHOLD - 0.5):
#### PRETRAINED:
- PASSES_1: 244 klatki, 489 detekcji ludzi (1 false positive - klatka nr 133, pozostałe true positive), 227 detekcji piłek (0 false positive, 17 false negative)
- PASSES_2: 342 klatki, 684 detekcje ludzi (wszystkie true positive), 235 detekcji piłek (0 false positive, 107 false negetive)
- PASSES_3: 298 klatek, 589 detekcji ludzi (7 połączonych detekcji - traktowane jako brak predykcji czyli 14 false negative, pozostałe true positive), 264 detekcje piłek (0 false positive, 34 false negative)
#### TRAINED:
- PASSES_1: 244 klatki, 488 detekcji ludzi (wszystkie true positive), 240 detekcji piłek (0 false positive, 4 false negative)
- PASSES_2: 342 klatki, 684 detekcje ludzi (wszystkie true positive), 265 detekcji piłek (0 false positive, 77 false negetive)
- PASSES_3: 298 klatek, 589 detekcji ludzi (7 połączonych detekcji - traktowane jako brak predykcji czyli 14 false negative, pozostałe true positive), 273 detekcje piłek (0 false positive, 25 false negative)

<br>

### WYNIKI - Algorithmic Detector (TRAINED YOLO CONFIDENCE THRESHOLD - 0.25):
- PASSES_1: 244 klatki, 244 detekcje pierwszej osoby, 244 detekcje drugiej osoby, 244 detekcje piłek
- PASSES_2: 342 klatki, 342 detekcje pierwszej osoby, 342 detekcje drugiej osoby, 283 detekcji piłek (59 false negative)
- PASSES_3: 298 klatek, 294 detekcje pierwszej osoby (4 false negative), 296 detekcji drugiej osoby (2 false negative), 280 detekcji piłek (18 false negative)
  
<br><br>

## TO-DO:
- DaSiam
