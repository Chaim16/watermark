
import os
import logging
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ImageUtils:
    """
    图像工具类，提供图像加载、保存等功能
    """
    
    @staticmethod
    def load_image(file_path: str) -> Optional[np.ndarray]:
        """
        加载图片文件
        
        Args:
            file_path: 图片文件路径
        
        Returns:
            OpenCV格式图像 (H, W, 3) BGR格式，失败返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None
            
            # 使用OpenCV加载图片
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if image is None:
                # 尝试使用PIL加载
                try:
                    pil_image = Image.open(file_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"无法加载图片 {file_path}: {str(e)}")
                    return None
            
            # 处理WebP格式
            if file_path.lower().endswith('.webp'):
                try:
                    pil_image = Image.open(file_path).convert('RGB')
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"加载WebP图片失败 {file_path}: {str(e)}")
                    return None
            
            logger.debug(f"成功加载图片: {file_path}, 尺寸: {image.shape}")
            return image
        
        except Exception as e:
            logger.error(f"加载图片失败 {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, file_path: str, quality: int = 95):
        """
        保存图片文件
        
        Args:
            image: OpenCV格式图像 (H, W, 3) BGR格式
            file_path: 输出文件路径
            quality: JPEG质量 (0-100)
        """
        try:
            # 获取文件扩展名
            ext = Path(file_path).suffix.lower()
            
            if ext in ['.jpg', '.jpeg']:
                # JPEG格式
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext == '.png':
                # PNG格式
                cv2.imwrite(file_path, image)
            elif ext == '.webp':
                # WebP格式
                # 使用PIL保存WebP
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                pil_image.save(file_path, 'WEBP', quality=quality)
            else:
                # 默认使用PNG格式
                cv2.imwrite(file_path, image)
            
            logger.debug(f"成功保存图片: {file_path}")
        
        except Exception as e:
            logger.error(f"保存图片失败 {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
        """
        调整图像大小，保持宽高比
        
        Args:
            image: 输入图像
            max_size: 最大尺寸（宽度或高度）
        
        Returns:
            调整后的图像
        """
        height, width = image.shape[:2]
        
        if max(width, height) <= max_size:
            return image.copy()
        
        # 计算缩放比例
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 使用双线性插值
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        将BGR图像转换为RGB
        
        Args:
            image: BGR格式图像
        
        Returns:
            RGB格式图像
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def convert_to_bgr(image: np.ndarray) -> np.ndarray:
        """
        将RGB图像转换为BGR
        
        Args:
            image: RGB格式图像
        
        Returns:
            BGR格式图像
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """
        获取图像信息
        
        Args:
            image: 输入图像
        
        Returns:
            图像信息字典
        """
        info = {
            'height': image.shape[0],
            'width': image.shape[1],
            'channels': image.shape[2] if image.ndim == 3 else 1,
            'dtype': str(image.dtype),
            'size_bytes': image.nbytes
        }
        return info
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        检查文件是否为图片文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否为图片文件
        """
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        ext = Path(file_path).suffix.lower()
        return ext in valid_extensions
    
    @staticmethod
    def normalize_mask(mask: np.ndarray) -> np.ndarray:
        """
        归一化mask到0-255范围
        
        Args:
            mask: 输入mask
        
        Returns:
            归一化后的mask
        """
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).astype(np.uint8)
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        return mask
