import cv2
import numpy as np

def safe_unsharp_mask(img, strength=1.0, radius=3):
    """安全的USM锐化，自动修正核大小"""
    # 确保核大小为奇数且≥3
    radius = max(1, int(radius))
    ksize = 2 * radius + 1  # 保证为奇数 (3,5,7...)

    # 高斯模糊（自动处理非法核大小）
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 锐化
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def high_quality_perspective_correction(img, points):
    # --------------- 预处理 ---------------
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # 去噪（确保输入为uint8）
    denoised = cv2.fastNlMeansDenoisingColored(
        img, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )

    # --------------- 透视变换 ---------------
    ordered_pts = order_points(points)
    # width = int(max(
    #     np.linalg.norm(ordered_pts[0] - ordered_pts[1]),
    #     np.linalg.norm(ordered_pts[2] - ordered_pts[3])
    # ))
    height = int(max(
        np.linalg.norm(ordered_pts[0] - ordered_pts[3]),
        np.linalg.norm(ordered_pts[1] - ordered_pts[2])
    ))
    width = int(height * 0.707)

    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

    # 使用浮点型提升精度
    corrected = cv2.warpPerspective(
        denoised.astype(np.float32),
        M,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT
    )

    # --------------- 后处理 ---------------
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    sharpened = safe_unsharp_mask(corrected, strength=1.2, radius=2)
    return sharpened


def order_points(pts):
    """角点排序逻辑保持不变"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect