import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from VehiclePlateRecognition import Instance


instance = Instance.from_file('test/sample.png')
result = instance.predict()
print(">> result", result)

instance.save_image('test/sample-output.png')
instance.save_plates_image('test/sample-output')
