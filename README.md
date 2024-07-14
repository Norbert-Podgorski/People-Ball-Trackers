# People-Ball-Trackers
<br>

#### SiamRPNBIG.model can be downloaded from [Google Drive.](https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H)

<br>

## NOTES
### RESULTS - YOLO (CONFIDENCE THRESHOLD - 0.5):
#### PRETRAINED:
PASSES_1: 244 frames, 489 human detections (1 false positive - frame no. 133, the rest true positives), 227 ball detections (17 false negatives) <br>
PASSES_2: 342 frames, 684 human detections (all true positives), 235 ball detections (107 false negatives) <br>
PASSES_3: 298 frames, 589 human detections (7 combined detections - treated as no predictions, thus 14 false negatives, the rest true positives), 264 ball detections (34 false negatives)

#### TRAINED:
PASSES_1: 244 frames, 488 human detections (all true positives), 240 ball detections (0 false positives, 4 false negatives) <br>
PASSES_2: 342 frames, 684 human detections (all true positives), 265 ball detections (0 false positives, 77 false negatives) <br>
PASSES_3: 298 frames, 589 human detections (7 combined detections - treated as no predictions, thus 14 false negatives, the rest true positives), 273 ball detections (25 false negatives)

<br>

### RESULTS - Algorithmic Detector (TRAINED YOLO CONFIDENCE THRESHOLD - 0.25):
PASSES_1: 244 frames, 244 detections of the first person, 244 detections of the second person, 244 ball detections <br>
PASSES_2: 342 frames, 342 detections of the first person, 342 detections of the second person, 283 ball detections (59 false negatives) <br>
PASSES_3: 298 frames, 294 detections of the first person (4 false negatives), 296 detections of the second person (2 false negatives), 280 ball detections (18 false negatives)

<br>

### RESULTS - DaSiamRPN Detector (TRAINED YOLO CONFIDENCE THRESHOLD - 0.25, SCORE THRESHOLD - 0.95):
PASSES_1: 244 frames, 244 detections of the first person, 244 detections of the second person, 244 ball detections <br>
PASSES_2: 342 frames, 342 detections of the first person, 342 detections of the second person, 320 ball detections (22 false negatives) <br>
PASSES_3: 298 frames, 294 detections of the first person (4 false negatives), 296 detections of the second person (2 false negatives), 293 ball detections (5 false negatives)

