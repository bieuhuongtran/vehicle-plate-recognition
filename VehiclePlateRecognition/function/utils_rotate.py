import cv2
import numpy as np
import math

# ===================== GÓC ĐÁY =====================
def angle_from_bottom_edge(quad):
    pts = quad.reshape(-1,2)
    idx = np.argsort(pts[:,1])[-2:]
    A, B = pts[idx[0]], pts[idx[1]]
    dy, dx = (B[1]-A[1]), (B[0]-A[0])
    return math.degrees(math.atan2(dy, dx))

# ===================== XOAY KHÔNG MẤT ẢNH =====================
def rotate_keep_all(img, angle_deg):
    (h, w) = img.shape[:2]
    center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

    cos = abs(M[0,0])
    sin = abs(M[0,1])

    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))

    M[0,2] += (nW/2) - center[0]
    M[1,2] += (nH/2) - center[1]

    return cv2.warpAffine(
        img, M, (nW,nH),
        flags=cv2.INTER_LINEAR,
        borderValue=(0,0,0)
    )
