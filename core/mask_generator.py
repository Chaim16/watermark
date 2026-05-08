
import numpy as np
import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MaskGenerator:
    """
    水印检测和Mask生成器
    使用多种图像处理技术自动检测水印区域
    """
    
    def __init__(self):
        pass
    
    def detect_watermark(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        自动检测水印区域并生成Mask
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
        
        Returns:
            Mask图像 (H, W) uint8格式，白色为水印区域
        """
        try:
            if image is None or image.ndim != 3:
                logger.error("无效的输入图像")
                return None
            
            # 尝试多种检测方法
            masks = []
            
            # 1. 边缘检测
            edge_mask = self._detect_by_edge(image)
            if edge_mask is not None and np.any(edge_mask > 0):
                masks.append(edge_mask)
            
            # 2. 亮度异常检测
            brightness_mask = self._detect_by_brightness(image)
            if brightness_mask is not None and np.any(brightness_mask > 0):
                masks.append(brightness_mask)
            
            # 3. 饱和度检测
            saturation_mask = self._detect_by_saturation(image)
            if saturation_mask is not None and np.any(saturation_mask > 0):
                masks.append(saturation_mask)
            
            # 4. 模板匹配（查找常见水印模式）
            template_mask = self._detect_by_template(image)
            if template_mask is not None and np.any(template_mask > 0):
                masks.append(template_mask)
            
            # 合并所有mask
            if masks:
                combined_mask = self._combine_masks(masks)
                refined_mask = self._refine_mask(combined_mask)
                return refined_mask
            else:
                logger.info("未检测到水印区域")
                return None
        
        except Exception as e:
            logger.error(f"水印检测失败: {str(e)}", exc_info=True)
            return None
    
    def _detect_by_edge(self, image: np.ndarray) -> np.ndarray:
        """
        使用边缘检测检测水印
        水印通常有明显的边界特征
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=2)
        
        return edges
    
    def _detect_by_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        使用亮度异常检测水印
        水印通常比周围区域更亮或更暗
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        
        # 计算亮度直方图
        hist = cv2.calcHist([value_channel], [0], None, [256], [0, 256])
        
        # 找到可能的水印亮度范围
        # 水印通常是小面积但高对比度的区域
        threshold = np.percentile(value_channel, 95)
        
        # 创建mask
        mask = (value_channel > threshold).astype(np.uint8) * 255
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        return mask
    
    def _detect_by_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        使用饱和度检测水印
        水印区域通常饱和度较低（灰色文字）
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv[:, :, 1]
        
        # 低饱和度区域可能是水印
        low_saturation = saturation_channel < 30
        
        # 创建mask
        mask = low_saturation.astype(np.uint8) * 255
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        return mask
    
    def _detect_by_template(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        使用模板匹配检测常见水印模式
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 尝试检测重复模式（常见于网站水印）
        # 使用FFT检测周期性模式
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift))
        
        # 查找高频分量（可能表示重复模式）
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        
        # 创建一个mask，只保留中心区域以外的高频分量
        high_freq_mask = np.ones_like(magnitude)
        radius = min(rows, cols) // 8
        cv2.circle(high_freq_mask, (center_col, center_row), radius, 0, -1)
        
        # 如果高频分量显著，可能存在水印模式
        high_freq_energy = np.sum(magnitude * high_freq_mask)
        
        if high_freq_energy > 1e7:
            # 使用自适应阈值
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
        
        return None
    
    def _combine_masks(self, masks: list) -> np.ndarray:
        """
        合并多个mask
        """
        combined = np.zeros_like(masks[0])
        
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        优化mask质量
        """
        # 去除小的孤立区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        refined = np.zeros_like(mask)
        
        # 只保留面积大于阈值的区域
        min_area = 50  # 最小面积阈值
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(refined, [contour], 0, 255, -1)
        
        # 扩展边界，确保完整覆盖水印
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_DILATE, kernel, iterations=1)
        
        return refined
    
    def generate_mask_from_box(self, image_shape: tuple, box: tuple) -> np.ndarray:
        """
        根据矩形框生成mask
        
        Args:
            image_shape: 图像形状 (H, W, ...)
            box: 矩形框 (x1, y1, x2, y2)
        
        Returns:
            Mask图像
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = box
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        mask[y1:y2, x1:x2] = 255
        
        return mask
