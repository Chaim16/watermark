
import os
import logging
import threading
from pathlib import Path
from typing import List
from PyQt5.QtCore import QObject, pyqtSignal

from core.image_utils import ImageUtils
from core.mask_generator import MaskGenerator
from core.inpaint import LaMaInpainter

logger = logging.getLogger(__name__)


class BatchProcessor(QObject):
    """
    批量处理器，支持多线程批量处理图片
    """
    
    progress_signal = pyqtSignal(int, int)
    completed_signal = pyqtSignal(int, int)
    error_signal = pyqtSignal(str)
    
    def __init__(self, inpainter: LaMaInpainter):
        super().__init__()
        self.inpainter = inpainter
        self.mask_generator = MaskGenerator()
        self.running = False
        self.success_count = 0
        self.fail_count = 0
    
    def process_batch(self, image_paths: List[Path], output_dir: str, auto_detect: bool = True):
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录
            auto_detect: 是否自动检测水印
        """
        self.running = True
        self.success_count = 0
        self.fail_count = 0
        self.total_count = len(image_paths)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用线程处理
        thread = threading.Thread(
            target=self._process_batch_thread,
            args=(image_paths, output_dir, auto_detect),
            daemon=True
        )
        thread.start()
    
    def _process_batch_thread(self, image_paths: List[Path], output_dir: str, auto_detect: bool):
        """
        批量处理线程
        """
        for i, image_path in enumerate(image_paths):
            if not self.running:
                break
            
            try:
                # 更新进度
                self.progress_signal.emit(i + 1, self.total_count)
                
                # 处理单张图片
                success = self._process_single_image(image_path, output_dir, auto_detect)
                
                if success:
                    self.success_count += 1
                else:
                    self.fail_count += 1
            
            except Exception as e:
                logger.error(f"处理 {image_path} 时发生错误: {str(e)}")
                self.fail_count += 1
                self.error_signal.emit(f"{image_path}: {str(e)}")
        
        # 完成
        self.running = False
        self.completed_signal.emit(self.success_count, self.fail_count)
    
    def _process_single_image(self, image_path: Path, output_dir: str, auto_detect: bool) -> bool:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            auto_detect: 是否自动检测水印
        
        Returns:
            是否处理成功
        """
        try:
            # 加载图片
            image = ImageUtils.load_image(str(image_path))
            if image is None:
                logger.error(f"无法加载图片: {image_path}")
                return False
            
            # 生成mask
            if auto_detect:
                mask = self.mask_generator.detect_watermark(image)
                if mask is None or mask.sum() == 0:
                    logger.warning(f"未检测到水印，跳过: {image_path}")
                    return False
            else:
                # 如果不自动检测，创建全白mask（修复整个图像）
                mask = None
            
            # 执行修复
            if mask is not None:
                result = self.inpainter.inpaint(image, mask)
            else:
                result = image.copy()
            
            if result is None:
                logger.error(f"修复失败: {image_path}")
                return False
            
            # 保存结果
            output_path = Path(output_dir) / f"{image_path.stem}_clean{image_path.suffix}"
            ImageUtils.save_image(result, str(output_path))
            
            logger.info(f"成功处理: {image_path} -> {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"处理 {image_path} 失败: {str(e)}")
            return False
    
    def stop(self):
        """
        停止批量处理
        """
        self.running = False
