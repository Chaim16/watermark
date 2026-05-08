
import os
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class LaMaInpainter:
    """
    LaMa图像修复器
    使用lama-cleaner库进行AI图像修复
    """
    
    def __init__(self):
        self.model_manager = None
        self.device = "cpu"
        self._init_model()
    
    def _init_model(self):
        """
        初始化LaMa模型
        """
        try:
            if not LAMA_AVAILABLE:
                logger.warning("lama-cleaner库未安装，将使用简单的OpenCV修复")
                return
            
            # 自动检测GPU
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("检测到CUDA设备，使用GPU加速")
            else:
                self.device = "cpu"
                logger.info("未检测到CUDA设备，使用CPU")
            
            # 初始化模型管理器
            self.model_manager = ModelManager(
                name="lama",
                device=self.device,
                no_half=False,
                offload=False
            )
            
            logger.info("LaMa模型初始化完成")
            
        except Exception as e:
            logger.error(f"初始化LaMa模型失败: {str(e)}")
            self.model_manager = None
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        执行图像修复
        
        Args:
            image: 原始图像 (H, W, 3) BGR格式
            mask: 修复掩码 (H, W) 0-255, 白色为需要修复的区域
        
        Returns:
            修复后的图像，如果失败返回None
        """
        try:
            # 验证输入
            if image is None or mask is None:
                logger.error("输入图像或掩码为空")
                return None
            
            if image.ndim != 3:
                logger.error("图像必须是3通道彩色图像")
                return None
            
            if mask.ndim != 2:
                logger.error("掩码必须是单通道图像")
                return None
            
            if image.shape[:2] != mask.shape[:2]:
                logger.error("图像和掩码尺寸不匹配")
                return None
            
            # 如果mask全黑，直接返回原图
            if mask.sum() == 0:
                logger.warning("掩码为空，直接返回原图")
                return image.copy()
            
            # 使用LaMa修复
            if self.model_manager is not None:
                return self._inpaint_lama(image, mask)
            else:
                # 降级使用OpenCV修复
                return self._inpaint_opencv(image, mask)
        
        except Exception as e:
            logger.error(f"修复失败: {str(e)}", exc_info=True)
            return None
    
    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        使用LaMa模型进行修复
        """
        # LaMa期望RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保mask是0-255的uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # 创建配置
        config = Config(
            image=image_rgb,
            mask=mask_uint8,
            config={
                "hd_strategy": "auto",
                "hd_strategy_crop_trigger_size": 1280,
                "hd_strategy_crop_size": 512,
                "hd_strategy_padding": 32,
            }
        )
        
        # 执行修复
        result = self.model_manager(image_rgb, mask_uint8, config)
        
        # 返回BGR格式
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    def _inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        使用OpenCV进行简单修复（降级方案）
        """
        # 确保mask是0-255的uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # 使用Telea算法进行修复
        result = cv2.inpaint(
            image,
            mask_uint8,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
        
        return result
    
    def cleanup(self):
        """
        清理资源
        """
        if self.model_manager is not None:
            try:
                # 释放模型占用的显存
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("已清理模型资源")
            except Exception as e:
                logger.error(f"清理资源失败: {str(e)}")
