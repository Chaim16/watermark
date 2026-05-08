import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPoint


class ImageLabel(QLabel):
    """
    自定义图片标签，支持绘制
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
    
    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.setFixedSize(pixmap.size())
        self.update()
    
    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.pixmap)


class ImageViewer(QWidget):
    """
    图片查看器组件，支持缩放、拖动、绘制Mask
    """
    
    mask_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, enable_drawing: bool = False):
        super().__init__()
        self.enable_drawing = enable_drawing
        
        # 图片数据
        self.original_image = None
        self.current_image = None
        self.mask = None
        self.mask_history = []
        
        # 缩放和拖动状态
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.dragging = False
        self.last_pos = QPoint(0, 0)
        
        # 画笔设置
        self.brush_size = 20
        self.brush_color = (255, 0, 0)  # 红色
        
        # 对比模式
        self.compare_mode = False
        self.compare_image = None
        self.compare_divider = 0.5
        
        # 初始化UI
        self._init_ui()
    
    def _init_ui(self):
        """
        初始化UI布局
        """
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scroll_area)
        
        # 图片显示部件
        self.image_label = ImageLabel()
        self.scroll_area.setWidget(self.image_label)
    
    def set_image(self, image: np.ndarray):
        """
        设置显示的图片
        """
        self.original_image = image.copy()
        self.current_image = image.copy()
        
        # 初始化mask
        if self.mask is None or self.mask.shape[:2] != image.shape[:2]:
            self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
            self.mask_history = []
        
        self._update_display()
    
    def set_mask(self, mask: np.ndarray):
        """
        设置Mask
        """
        if mask.shape[:2] == self.original_image.shape[:2]:
            self.mask = mask.copy()
            self.mask_history.append(mask.copy())
            self.mask_updated.emit(self.mask)
            self._update_display()
    
    def clear_mask(self):
        """
        清空Mask
        """
        if self.original_image is not None:
            self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.mask_history = []
            self.mask_updated.emit(self.mask)
            self._update_display()
    
    def set_brush_size(self, size: int):
        """
        设置画笔大小
        """
        self.brush_size = size
    
    def undo_last_stroke(self):
        """
        撤销最后一笔
        """
        if self.mask_history:
            self.mask_history.pop()
            if self.mask_history:
                self.mask = self.mask_history[-1].copy()
            else:
                self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.mask_updated.emit(self.mask)
            self._update_display()
    
    def set_compare_images(self, before: np.ndarray, after: np.ndarray):
        """
        设置对比模式的两张图片
        """
        self.compare_mode = True
        self.original_image = before
        self.compare_image = after
        self._update_display()
    
    def _update_display(self):
        """
        更新显示内容
        """
        if self.original_image is None:
            return
        
        if self.compare_mode and self.compare_image is not None:
            self._display_compare()
        else:
            self._display_with_mask()
    
    def _display_with_mask(self):
        """
        显示图片和Mask叠加效果
        """
        # 创建带Mask的显示图像
        display_image = self.original_image.copy()
        
        # 如果有mask，添加红色半透明覆盖
        if self.mask is not None and np.any(self.mask > 0):
            mask_indices = self.mask > 0
            display_image[mask_indices] = (
                display_image[mask_indices] * 0.5 + 
                np.array([255, 0, 0]) * 0.5
            ).astype(np.uint8)
        
        # 转换为QPixmap
        pixmap = self._cv_to_pixmap(display_image)
        
        # 设置到显示部件
        self.image_label.set_pixmap(pixmap)
    
    def _display_compare(self):
        """
        显示对比模式（左右分割）
        """
        if self.original_image is None or self.compare_image is None:
            return
        
        # 确保两张图片尺寸相同
        if self.original_image.shape != self.compare_image.shape:
            return
        
        height, width = self.original_image.shape[:2]
        divider_x = int(width * self.compare_divider)
        
        # 创建对比图像
        compare_img = self.original_image.copy()
        compare_img[:, divider_x:] = self.compare_image[:, divider_x:]
        
        # 添加分割线
        cv2.line(compare_img, (divider_x, 0), (divider_x, height), (255, 255, 255), 2)
        
        # 转换为QPixmap
        pixmap = self._cv_to_pixmap(compare_img)
        
        self.image_label.set_pixmap(pixmap)
    
    def _cv_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """
        将OpenCV图像转换为QPixmap
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # BGR转RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        return QPixmap.fromImage(q_image)
    
    def mousePressEvent(self, event):
        """
        鼠标按下事件
        """
        if not self.enable_drawing or self.original_image is None:
            return
        
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.last_pos = self._get_image_coords(event.pos())
            self._draw_at_position(self.last_pos)
    
    def mouseMoveEvent(self, event):
        """
        鼠标移动事件
        """
        if self.original_image is None:
            return
        
        current_pos = self._get_image_coords(event.pos())
        
        if self.enable_drawing and event.buttons() & Qt.LeftButton:
            self._draw_line(self.last_pos, current_pos)
            self.last_pos = current_pos
        elif event.buttons() & Qt.RightButton:
            # 右键拖动
            delta = current_pos - self.last_pos
            self.offset += delta
            self.last_pos = current_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
    
    def mouseReleaseEvent(self, event):
        """
        鼠标释放事件
        """
        if self.enable_drawing and self.mask is not None:
            # 保存当前mask到历史记录
            self.mask_history.append(self.mask.copy())
            self.mask_updated.emit(self.mask)
    
    def wheelEvent(self, event):
        """
        滚轮缩放事件
        """
        if self.original_image is None:
            return
        
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        
        # 限制缩放范围
        new_scale = self.scale * zoom_factor
        if 0.1 <= new_scale <= 5.0:
            self.scale = new_scale
            # TODO: 实现缩放功能
            event.accept()
    
    def _get_image_coords(self, widget_pos: QPoint) -> QPoint:
        """
        将部件坐标转换为图像坐标
        """
        # 获取滚动区域的偏移
        h_offset = self.scroll_area.horizontalScrollBar().value()
        v_offset = self.scroll_area.verticalScrollBar().value()
        
        return QPoint(
            int((widget_pos.x() + h_offset) / self.scale),
            int((widget_pos.y() + v_offset) / self.scale)
        )
    
    def _draw_at_position(self, pos: QPoint):
        """
        在指定位置绘制
        """
        if self.mask is None:
            return
        
        h, w = self.mask.shape
        
        # 确保坐标在图像范围内
        x, y = pos.x(), pos.y()
        if x < 0 or x >= w or y < 0 or y >= h:
            return
        
        # 绘制圆形画笔痕迹
        radius = self.brush_size // 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    distance = (dx**2 + dy**2)**0.5
                    if distance <= radius:
                        self.mask[ny, nx] = 255
        
        self._update_display()
    
    def _draw_line(self, start: QPoint, end: QPoint):
        """
        绘制线段
        """
        if self.mask is None:
            return
        
        # 使用Bresenham算法绘制线段
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        while True:
            self._draw_at_position(QPoint(x1, y1))
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x1 += sx
            
            if e2 < dx:
                err += dx
                y1 += sy
    
    def clear(self):
        """
        清空显示
        """
        self.original_image = None
        self.current_image = None
        self.mask = None
        self.mask_history = []
        self.image_label.clear()
        self.update()
