import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import VehiclePlateRecognition
with open('test/sample-1.jpg', 'rb') as file:
    imgBuffer = file.read()
    result = VehiclePlateRecognition.recognizeLicensePlateBuffer(imgBuffer)
    print(result)