import cv2
import numpy as np
import hqpc
import whiteBalanced

def preprocess_image(img):
    # 转为灰度图并降噪
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5, 0)
    # 自适应阈值处理增强边缘
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey()
    return thresh

def find_document_contour(img, original_img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # 按面积降序排序，保留前10个候选
    sorted_contours = sorted(
        zip(contours, hierarchy[0]),
        key=lambda x: cv2.contourArea(x[0]),
        reverse=True
    )[:10]
    for cnt, hier in sorted_contours:
        # 要求是外层轮廓（父轮廓为-1）
        if hier[3] != -1:
            continue

        # 近似为四边形
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # 检查面积占比（至少占图像面积的20%）
        img_area = original_img.shape[0] * original_img.shape[1]
        cnt_area = cv2.contourArea(approx)
        if cnt_area / img_area < 0.2:
            continue

        return approx.reshape(4, 2)
    return None

def rectification(img):
    # inputFile = "input02.jpg"
    # img = cv2.imread(inputFile, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    processed = preprocess_image(img)
    contour = find_document_contour(processed, img)

    if contour is not None:
        # 绘制检测到的角点
        for (x, y) in contour:
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)
        # cv2.imshow('Detected Corners', img)
        # cv2.waitKey(0)

        # 透视校正
        corrected = hqpc.high_quality_perspective_correction(img, contour)
        # outputFile = inputFile.split('.')[0] + '_rec.jpg'
        # cv2.imwrite('output/' + outputFile, corrected)
        wb = whiteBalanced.white_balance(corrected)
        # outputFile = inputFile.split('.')[0] + '_wb.jpg'
        # cv2.imwrite('output/' + outputFile, wb)
        return wb
    else:
        print("未检测到文件边界")