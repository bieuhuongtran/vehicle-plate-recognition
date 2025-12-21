import cv2
import numpy as np
import math

# ===================== BINARY =====================
def binarize_for_contours(channel):
    blur = cv2.GaussianBlur(channel, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
        iterations=1
    )
    return th

# ===================== CONTOURS → QUADS =====================
def approx_quads_from_contours(contours, min_perimeter=50):
    quads = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < min_perimeter:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4,2).astype(np.float32))
    return quads

# ===================== ORDER TL–TR–BR–BL =====================
def order_quad_points(pts4):
    pts = np.array(pts4, dtype=np.float32).reshape(-1,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

# ===================== SCORE CHỌN BIỂN =====================
def score_plate_quad(quad, img_area):
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    if w == 0 or h == 0:
        return -1
    aspect = w / float(h)
    area = w * h
    area_frac = area / float(img_area)

    # Ràng buộc hợp lý
    if not (1.0 <= aspect <= 7.0):
        return -1
    if not (0.001 <= area_frac <= 0.40):
        return -1

    score = area * (1.0 - abs(aspect - 2.5)/4.0)
    return score

# ===================== CHỌN 4-POINT WARP =====================
def four_point_transform(image, quad, force_width=None, force_ar=None):
    rect = order_quad_points(quad)
    (tl, tr, br, bl) = rect

    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    W = int(max(widthA, widthB))
    H = int(max(heightA, heightB))

    if force_width is not None:
        W = int(force_width)
    if force_ar is not None and force_ar > 0:
        H = max(1, int(W / float(force_ar)))

    W = max(W, 1); H = max(H, 1)

    dst = np.array([
        [0,0], [W-1,0], [W-1,H-1], [0,H-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped
