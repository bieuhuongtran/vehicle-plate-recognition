import os
import torch

current_directory = os.path.dirname(__file__)
yolo_LP_detect = torch.hub.load(f'{current_directory}/yolov5', 'custom', path=f'{current_directory}/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load(f'{current_directory}/yolov5', 'custom', path=f'{current_directory}/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

