# Face-Recognition-on-RaspberryPi
Face detection and face recognition based on raspberry pi 3. I use raspbian stretch with python 3.5, but other OS should also be good.

## How to run?
- Run the 'trainModel.py' onece on your *computer*.
- Plant the trained model to your raspberry pi and run the 'faceRecognize.py' to realize real-time face recognition.

## Notes
This project uses LBPH trainer in OpenCV 3+, which means OpenCV 2 is not supported.

To install OpenCV 3 on your raspberry pi, I strongly recommend you to see this [tutorial](https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/).
