import cv2
import numpy as np


def binarize_for_contours(channel):
    """binarize channel (8-bit) for findContours."""
    blur = cv2.GaussianBlur(channel, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return th


def approx_quads_from_contours(contours):
    """Returns an approximate list of quadrilaterals (4x2) from contours."""
    quads = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 50:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))
    return quads


def order_quad_points(pts4):
    """Arrange 4 points: tl: top left, tr: top right, br: bottom right, bl: bottom left."""
    pts = np.array(pts4, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def score_plate_quad(quad, img_area):
    """Scoring based on box area and appropriate w/h ratio."""
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    if w == 0 or h == 0:
        return -1
    aspect = w / float(h)
    area = w * h
    area_frac = area / float(img_area)
    if not (1.0 <= aspect <= 7.0):
        return -1
    if not (0.001 <= area_frac <= 0.40):
        return -1
    score = area * (1.0 - abs(aspect - 2.5) / 4.0)
    return score


# === NEW: 4-Point perspective warp ===
def four_point_transform(image, quad, force_width=None, force_ar=None):
    """
    quad: 4x2 (tl, tr, br, bl)
    force_width: force the output width (px)
    force_ar: force the ratio W/H (E.g. 4.0 for horizontal plate)
    """
    rect = order_quad_points(quad)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    W = int(max(widthA, widthB))
    H = int(max(heightA, heightB))
    if force_width is not None:
        W = int(force_width)
    if force_ar is not None and force_ar > 0:
        H = max(1, int(W / float(force_ar)))

    W = max(W, 1)
    H = max(H, 1)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W, H))


def normalize_plate(image):
    h, w = image.shape[:2]
    img_area = w * h

    quads = []
    blue, green, red = cv2.split(image)

    # Red
    th_r = binarize_for_contours(red)
    contours_r, _ = cv2.findContours(th_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    quads += approx_quads_from_contours(contours_r)

    # Green
    th_g = binarize_for_contours(green)
    contours_g, _ = cv2.findContours(th_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    quads += approx_quads_from_contours(contours_g)

    # Blue
    th_b = binarize_for_contours(blue)
    contours_b, _ = cv2.findContours(th_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    quads += approx_quads_from_contours(contours_b)

    best_quad = None
    best_score = -1
    for q in quads:
        q_ord = order_quad_points(q)
        sc = score_plate_quad(q_ord, img_area)
        if sc > best_score:
            best_score = sc
            best_quad = q_ord

    # Fallback: If there is no quadrilateral with four vertices, use minAreaRect from the "best" channel based on total area.
    if best_quad is None:
        candidates = sorted(
            [("b", contours_b), ("g", contours_g), ("r", contours_r)],
            key=lambda x: sum(cv2.contourArea(c) for c in x[1]),
            reverse=True,
        )
        cnts = candidates[0][1] if candidates else contours_b
        if not cnts:
            # raise RuntimeError("Không tìm thấy contour nào đủ lớn.")
            return None
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        best_quad = order_quad_points(box)

    # 190/140 : Standard ratio for motorcycle license plates
    warped = four_point_transform(image, best_quad, force_width=None, force_ar=190 / 140)

    # If the image is still vertical > horizontal after the warp, rotate it 90°
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped
