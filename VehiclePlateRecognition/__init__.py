import math
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from .model.yolo import yolo_LP_detect
from .rotate import deskew_plate
from .normalize import normalize_plate
from .ocr import ocr_plate


class Plate:
    def __init__(self, img, x1, y1, x2, y2):
        self._text = None

        self.img = img
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def save_img(self, outputPath):
        Image.fromarray(self.img, "RGB").save(outputPath)

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    def ocr(self):
        crop_img = self.img
        lp = None
        for cc in range(0, 2):
            for ct in range(0, 2):
                dPlate = deskew_plate(crop_img, cc, ct)
                nPlate = normalize_plate(dPlate)
                lp = ocr_plate(nPlate)
                if lp:
                    return lp
        return None

    @property
    def text(self):
        if not self._text:
            self._text = self.ocr()
        return self._text

    def __repr__(self):
        return f"Plate (x1:{self.x1}, y1:{self.y1}, x2:{self.x2}, y2:{self.y2})"

    def predict(self):
        return {
            "x": self.x1,
            "y": self.y1,
            "w": self.w,
            "h": self.h,
            "text": self.text,
        }


class Instance:
    def __init__(self, source):
        self._plates = None

        self.source = source
        pil = Image.open(BytesIO(source)).convert("RGB")
        self.img = np.array(pil, dtype=np.uint8)

    def detect_plates(self):
        plates = []
        for result in (yolo_LP_detect(self.img, size=640).pandas().xyxy[0].values.tolist()):
            x1, y1, x2, y2, *_ = result
            plates.append(Plate(
                self.img[int(y1 - 10) : int(y2 + 10), int(x1 - 10) : int(x2 + 10)],# crop plate from source image
                x1, y1, x2, y2,
            ))
        return plates

    @property
    def plates(self) -> list[Plate]:
        if not self._plates:
            self._plates = self.detect_plates()
        return self._plates

    def save_image(self, outputPath):
        dbgImg = self.img.copy()
        for plate in self.plates:
            cv2.rectangle(
                dbgImg,
                (math.floor(plate.x1), math.floor(plate.y1 - 1)),
                (math.ceil(plate.x2 + 1), math.ceil(plate.y2 + 1)),
                (255, 0, 0),
                4,
            )
        cv2.imwrite(outputPath, dbgImg)

    def save_plates_image(self, outputPathPrefix):
        id = 0
        for plate in self.plates:
            id = id + 1
            plate.save_img(outputPathPrefix + "-plate-" + str(id) + ".png")

    def predict(self):
        results = []
        for plate in self.plates:
            plateResult = plate.predict()
            if plateResult["text"]:
                results.append(plateResult)
        return results

    @staticmethod
    def process_img_buffer(source):
        obj = Instance(source)
        return obj.predict()

    @staticmethod
    def from_file(input_file):
        with open(input_file, "rb") as file:
            return Instance(file.read())


def recognizeLicensePlateBuffer(source):
    return Instance.process_img_buffer(source)
