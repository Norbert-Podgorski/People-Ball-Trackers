# People-Ball-Trackers

A system that detects and tracks the movement of soccer players and the ball using the YOLO and DaSiamRPN networks, as well as the principles of motion kinematics.

The created system consists of the following modules:
- <b>Pretrained YOLO Detector:</b> uses a pre-trained YOLO network, used as a reference point
- <b>Trained YOLO Detector:</b> uses the YOLO network trained independently on a specially selected dataset
- <b>Algorithmic Detector:</b> is based on YOLO detections, additionally uses the principles of motion kinematics
- <b>DaSiamRPN Tracker:</b> uses the DaSiamRPN siamese network

These modules are connected as follows: <br>
![image](https://github.com/user-attachments/assets/75a4eaf6-2234-4354-b372-847afceb46d9)

Additionally, the following design patterns were used to create the modules:
- Factory
- Strategy
- State

<br>

The results can be seen in the visualizations folder.
SiamRPNBIG.model can be downloaded from [Google Drive.](https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H)
