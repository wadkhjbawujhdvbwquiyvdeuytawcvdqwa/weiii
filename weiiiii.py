import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt


def load_and_preprocess_images(ir_path, visible_path):
    """加载并预处理红外和可见光图像"""
    # 读取图像 - 红外图为灰度，可见光图为彩色
    ir_img = cv2.imread(ir_path, 0)  # 灰度图
    visible_img = cv2.imread(visible_path)  # 彩色图(BGR格式)

    # 确保图像尺寸一致
    if ir_img.shape != visible_img.shape[:2]:
        visible_img = cv2.resize(visible_img, (ir_img.shape[1], ir_img.shape[0]))

    # 归一化处理
    ir_img = ir_img.astype(np.float32) / 255.0
    visible_img = visible_img.astype(np.float32) / 255.0

    return ir_img, visible_img


def wavelet_decomposition(img, wavelet='db1', level=3):
    """对图像进行小波分解"""
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    return coeffs


def fusion_rule(coeffs_ir, coeffs_visible, rule='max'):
    """融合规则实现"""
    fused_coeffs = []

    # 处理近似系数(低频部分)
    if rule == 'average':
        fused_coeffs.append((coeffs_ir[0] + coeffs_visible[0]) / 2)
    elif rule == 'max':
        fused_coeffs.append(np.maximum(coeffs_ir[0], coeffs_visible[0]))
    elif rule == 'weighted':
        # 可根据具体应用场景设计权重
        weight_ir = 0.6
        weight_visible = 0.4
        fused_coeffs.append(weight_ir * coeffs_ir[0] + weight_visible * coeffs_visible[0])

    # 处理细节系数(高频部分)
    for i in range(1, len(coeffs_ir)):
        cH_ir, cV_ir, cD_ir = coeffs_ir[i]
        cH_visible, cV_visible, cD_visible = coeffs_visible[i]

        # 水平细节系数融合
        if rule == 'max':
            cH_fused = np.maximum(cH_ir, cH_visible)
        elif rule == 'weighted':
            # 可以根据能量或其他特征计算权重
            energy_ir = cH_ir ** 2
            energy_visible = cH_visible ** 2
            weight_ir = energy_ir / (energy_ir + energy_visible + 1e-10)
            weight_visible = 1 - weight_ir
            cH_fused = weight_ir * cH_ir + weight_visible * cH_visible

        # 垂直细节系数融合
        if rule == 'max':
            cV_fused = np.maximum(cV_ir, cV_visible)
        elif rule == 'weighted':
            energy_ir = cV_ir ** 2
            energy_visible = cV_visible ** 2
            weight_ir = energy_ir / (energy_ir + energy_visible + 1e-10)
            weight_visible = 1 - weight_ir
            cV_fused = weight_ir * cV_ir + weight_visible * cV_visible

        # 对角线细节系数融合
        if rule == 'max':
            cD_fused = np.maximum(cD_ir, cD_visible)
        elif rule == 'weighted':
            energy_ir = cD_ir ** 2
            energy_visible = cD_visible ** 2
            weight_ir = energy_ir / (energy_ir + energy_visible + 1e-10)
            weight_visible = 1 - weight_ir
            cD_fused = weight_ir * cD_ir + weight_visible * cD_visible

        fused_coeffs.append((cH_fused, cV_fused, cD_fused))

    return fused_coeffs


def wavelet_reconstruction(coeffs, wavelet='db1'):
    """对融合后的系数进行重构"""
    fused_img = pywt.waverec2(coeffs, wavelet)
    return fused_img


def ir_visible_fusion(ir_path, visible_path, wavelet='db1', level=3, rule='weighted'):
    """红外与可见光图像融合主函数"""
    # 加载并预处理图像
    ir_img, visible_img = load_and_preprocess_images(ir_path, visible_path)

    # 小波分解 - 红外图像
    coeffs_ir = wavelet_decomposition(ir_img, wavelet, level)

    # 对可见光图像的每个通道进行小波分解
    coeffs_visible_b = wavelet_decomposition(visible_img[:, :, 0], wavelet, level)
    coeffs_visible_g = wavelet_decomposition(visible_img[:, :, 1], wavelet, level)
    coeffs_visible_r = wavelet_decomposition(visible_img[:, :, 2], wavelet, level)

    # 分别融合每个通道的系数
    fused_coeffs_b = fusion_rule(coeffs_ir, coeffs_visible_b, rule)
    fused_coeffs_g = fusion_rule(coeffs_ir, coeffs_visible_g, rule)
    fused_coeffs_r = fusion_rule(coeffs_ir, coeffs_visible_r, rule)

    # 图像重构
    fused_b = wavelet_reconstruction(fused_coeffs_b, wavelet)
    fused_g = wavelet_reconstruction(fused_coeffs_g, wavelet)
    fused_r = wavelet_reconstruction(fused_coeffs_r, wavelet)

    # 裁剪到原始尺寸
    fused_b = fused_b[:ir_img.shape[0], :ir_img.shape[1]]
    fused_g = fused_g[:ir_img.shape[0], :ir_img.shape[1]]
    fused_r = fused_r[:ir_img.shape[0], :ir_img.shape[1]]

    # 归一化到[0,1]范围
    fused_b = np.clip(fused_b, 0, 1)
    fused_g = np.clip(fused_g, 0, 1)
    fused_r = np.clip(fused_r, 0, 1)

    # 合并三个通道为彩色图像
    fused_img = np.stack([fused_b, fused_g, fused_r], axis=2)

    # 转回8位图像
    fused_img = (fused_img * 255).astype(np.uint8)
    ir_img = (ir_img * 255).astype(np.uint8)
    visible_img = (visible_img * 255).astype(np.uint8)

    return ir_img, visible_img, fused_img


def evaluate_fusion(ir_img, visible_img, fused_img):
    """评估融合图像质量"""

    # 计算信息熵 (信息量)
    def entropy(img):
        if len(img.shape) == 3:  # 彩色图像
            entropy_sum = 0
            for i in range(3):
                hist = cv2.calcHist([img[:, :, i]], [0], None, [256], [0, 256])
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # 避免log(0)
                entropy_sum += -np.sum(hist * np.log2(hist))
            return entropy_sum / 3  # 返回平均熵
        else:  # 灰度图像
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # 避免log(0)
            return -np.sum(hist * np.log2(hist))

    # 计算互信息 (与源图像的相关性)
    def mutual_information(img1, img2):
        # 如果是彩色图像，转换为灰度
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1

        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2

        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
        hist12 = np.zeros((256, 256))

        for i in range(img1_gray.shape[0]):
            for j in range(img1_gray.shape[1]):
                hist12[img1_gray[i, j], img2_gray[i, j]] += 1

        hist12 = hist12 / hist12.sum()
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        mi = 0
        for i in range(256):
            for j in range(256):
                if hist12[i, j] > 0:
                    mi += hist12[i, j] * np.log2(hist12[i, j] / (hist1[i] * hist2[j]))

        return mi

    # 计算标准差 (对比度)
    if len(fused_img.shape) == 3:
        std = np.mean([np.std(fused_img[:, :, i]) for i in range(3)])
    else:
        std = np.std(fused_img)

    # 计算清晰度 (梯度平均值)
    if len(fused_img.shape) == 3:
        sobelx = cv2.Sobel(cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
    else:
        sobelx = cv2.Sobel(fused_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(fused_img, cv2.CV_64F, 0, 1, ksize=3)

    clarity = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2))

    # 计算与源图像的互信息
    mi_ir = mutual_information(ir_img, fused_img)
    mi_visible = mutual_information(visible_img, fused_img)

    # 计算信息熵
    ent = entropy(fused_img)

    return {
        "Standard Deviation": std,
        "Clarity": clarity,
        "Mutual Information with IR": mi_ir,
        "Mutual Information with Visible": mi_visible,
        "Entropy": ent
    }


def visualize_results(ir_img, visible_img, fused_img, metrics=None):
    """可视化结果"""
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    if len(ir_img.shape) == 2:  # 灰度图
        plt.imshow(ir_img, cmap='gray')
    else:  # 彩色图
        plt.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    plt.title('hongwai')
    plt.axis('off')

    plt.subplot(132)
    if len(visible_img.shape) == 2:  # 灰度图
        plt.imshow(visible_img, cmap='gray')
    else:  # 彩色图
        plt.imshow(cv2.cvtColor(visible_img, cv2.COLOR_BGR2RGB))
    plt.title('kejianguang')
    plt.axis('off')

    plt.subplot(133)
    if len(fused_img.shape) == 2:  # 灰度图
        plt.imshow(fused_img, cmap='gray')
    else:  # 彩色图
        plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    plt.title('ronghe')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 显示评估指标
    if metrics:
        print("\nFusion Quality Metrics:")
        for key, value in metrics.items():
            # 检查值是否为NumPy数组
            if isinstance(value, np.ndarray):
                # 如果是数组，取平均值
                print(f"{key}: {np.mean(value):.4f}")
            else:
                # 普通浮点数直接格式化
                print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    # 替换为实际图像路径
    ir_path = r"E:\Keshe\Hongwai\Aaa.jpg"
    visible_path = r"E:\Keshe\Keshihua\Bbb.jpg"

    # 执行融合
    ir_img, visible_img, fused_img = ir_visible_fusion(
        ir_path, visible_path,
        wavelet='db1',  # 可以尝试其他小波基，如 'haar', 'sym4', 'coif3' 等
        level=3,  # 分解层数
        rule='weighted'  # 融合规则: 'max', 'average', 'weighted'
    )

    # 评估融合质量
    metrics = evaluate_fusion(ir_img, visible_img, fused_img)

    # 可视化结果
    visualize_results(ir_img, visible_img, fused_img, metrics)

    # 保存融合图像
    cv2.imwrite("fused_image.jpg", fused_img)
