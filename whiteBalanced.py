import cv2
import numpy as np

def convert_channel_to_uint8(channel):
    """将任意范围的 float64 通道转换为 uint8"""
    min_val = np.min(channel)
    max_val = np.max(channel)
    # 归一化到 [0, 255]
    normalized = 255 * (channel - min_val) / (max_val - min_val + 1e-6)
    return np.clip(normalized, 0, 255).astype(np.uint8)

def adjust_white_balance(image, sat_scale=1.5, contrast_alpha=1.2, red_gain=2.0, red_sat_boost=2.0, protect_highlights=True):
    # 转换为灰度图并进行二值化处理以分割背景
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学操作去除噪声并增强背景区域
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 反转掩膜以获取背景区域
    background_mask = cv2.bitwise_not(thresh)

    # 计算背景区域的平均颜色
    mean_color = cv2.mean(image, mask=background_mask)
    mean_b, mean_g, mean_r = mean_color[:3]

    # 计算各通道增益，将背景调整为白色
    gain_b = 255.0 / mean_b if mean_b != 0 else 1.0
    gain_g = 255.0 / mean_g if mean_g != 0 else 1.0
    gain_r = 255.0 / mean_r if mean_r != 0 else 1.0

    # 应用增益调整并限制像素值范围
    balanced_image = image.copy().astype(np.float32)
    balanced_image[..., 0] = np.clip(balanced_image[..., 0] * gain_b, 0, 255)
    balanced_image[..., 1] = np.clip(balanced_image[..., 1] * gain_g, 0, 255)
    balanced_image[..., 2] = np.clip(balanced_image[..., 2] * gain_r, 0, 255)
    balanced_image = balanced_image.astype(np.uint8)

    # ---------- 修复后的饱和度增强 ----------
    hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 安全缩放饱和度通道
    s = cv2.convertScaleAbs(s, alpha=sat_scale, beta=0)

    # 显式确保数据类型为uint8
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    balanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. 提升对比度（线性变换）
    balanced_image = cv2.convertScaleAbs(balanced_image, alpha=contrast_alpha, beta=0)

    # 3. 锐化处理（增强边缘清晰度）
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    balanced_image = cv2.filter2D(balanced_image, -1, sharpen_kernel)
    return balanced_image

def enhance_red_saturation(image, sat_gain=5.0, red_hue_threshold=15):
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 定义红色色调范围（OpenCV的H范围为0-180）
    # 低红色区域（0到阈值）
    lower_red = (0, 50, 50)
    upper_red = (red_hue_threshold, 255, 255)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # 高红色区域（对称的180附近）
    lower_red = (180 - red_hue_threshold, 50, 50)
    upper_red = (180, 255, 255)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # 合并红色区域掩膜
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 仅对红色区域增强饱和度
    s = s.astype(np.float32)  # 转换为浮点型以进行乘法
    s[red_mask > 0] = np.clip(s[red_mask > 0] * sat_gain, 0, 255)
    s = s.astype(np.uint8)  # 转回uint8

    # 合并通道并转回BGR
    enhanced_hsv = cv2.merge([h, s, v])
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return result

def white_balance(img):
    output_image = adjust_white_balance(img)
    output_image = enhance_red_saturation(output_image)
    return output_image

if __name__ == '__main__':
    # 读取输入图像
    input_image = cv2.imread('output/input01_rec.jpg')

    # 调整白平衡
    output_image = adjust_white_balance(input_image)
    output_image = enhance_red_saturation(output_image)

    # 保存并显示结果
    cv2.imwrite('output1.jpg', output_image)
    # cv2.imshow('Adjusted Image', output_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()