# Face-Recognition-on-RaspberryPi
Face detection and face recognition based on raspberry pi 3. I use raspbian stretch with python 3.5, but other OS should also be good.

## How to run?
1. Run the 'trainModel.py' onece on your *computer*.
2. Plant the trained model to your raspberry pi and run the 'faceRecognize.py' to realize real-time face recognition.

## Notes
1. This project uses LBPH trainer in OpenCV 3+, which means OpenCV 2 is not supported.

&nbsp; &nbsp; &nbsp; To install OpenCV 3 on your raspberry pi, I strongly recommend you to see this [tutorial](https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/).

2. `Picamera` library is not used, which means you can debug/run the whole project on your computer.
