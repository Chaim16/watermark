
import os
import logging
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel, QProgressBar,
    QSlider, QSpinBox, QToolBar, QAction, QStatusBar,
    QTabWidget, QSplitter, QMessageBox, QDockWidget,
    QTextEdit
)
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

from ui.image_viewer import ImageViewer
from core.batch_processor import BatchProcessor
from core.mask_generator import MaskGenerator
from core.inpaint import LaMaInpainter
from core.image_utils import ImageUtils

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    主窗口类，管理整个应用界面
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI图片去水印工具")
        self.setMinimumSize(1200, 800)
        
        # 当前处理的图片路径
        self.current_image_path: Optional[Path] = None
        self.current_image = None
        self.current_mask = None
        self.result_image = None
        
        # 图片队列
        self.image_queue: List[Path] = []
        self.current_queue_index = 0
        
        # 初始化核心组件
        self._init_core_components()
        
        # 初始化UI
        self._init_ui()
        
        # 初始化状态栏
        self._init_status_bar()
        
        # 初始化快捷键
        self._init_shortcuts()
        
        logger.info("主窗口初始化完成")
    
    def _init_core_components(self):
        """
        初始化核心处理组件
        """
        self.lama_inpainter = LaMaInpainter()
        self.mask_generator = MaskGenerator()
        self.batch_processor = BatchProcessor(self.lama_inpainter)
        self.batch_processor.progress_signal.connect(self._update_progress)
        self.batch_processor.completed_signal.connect(self._on_batch_complete)
        self.batch_processor.error_signal.connect(self._on_batch_error)
    
    def _init_ui(self):
        """
        初始化用户界面布局
        """
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 工具栏
        self._init_toolbar()
        
        # 主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # 左侧：原图和Mask编辑区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 原图标签
        self.original_label = QLabel("原图")
        self.original_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_label)
        
        # 原图查看器（支持绘制mask）
        self.original_viewer = ImageViewer(enable_drawing=True)
        self.original_viewer.mask_updated.connect(self._on_mask_updated)
        left_layout.addWidget(self.original_viewer)
        
        # Mask编辑工具
        mask_tool_layout = QHBoxLayout()
        
        # 画笔大小
        brush_label = QLabel("画笔大小:")
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 100)
        self.brush_size_spin.setValue(20)
        self.brush_size_spin.valueChanged.connect(self._update_brush_size)
        
        # 撤销按钮
        undo_btn = QPushButton("撤销")
        undo_btn.clicked.connect(self.original_viewer.undo_last_stroke)
        
        # 清空按钮
        clear_btn = QPushButton("清空Mask")
        clear_btn.clicked.connect(self._clear_mask)
        
        mask_tool_layout.addWidget(brush_label)
        mask_tool_layout.addWidget(self.brush_size_spin)
        mask_tool_layout.addWidget(undo_btn)
        mask_tool_layout.addWidget(clear_btn)
        
        left_layout.addLayout(mask_tool_layout)
        main_splitter.addWidget(left_widget)
        
        # 右侧：修复结果区
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 结果标签
        self.result_label = QLabel("修复结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_label)
        
        # 结果查看器
        self.result_viewer = ImageViewer(enable_drawing=False)
        right_layout.addWidget(self.result_viewer)
        
        # 对比模式按钮
        compare_btn = QPushButton("开启对比模式")
        compare_btn.clicked.connect(self._toggle_compare_mode)
        right_layout.addWidget(compare_btn)
        
        main_splitter.addWidget(right_widget)
        
        # 设置分割器比例
        main_splitter.setSizes([600, 600])
        
        # 底部日志面板
        self._init_log_panel()
    
    def _init_toolbar(self):
        """
        初始化顶部工具栏
        """
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)
        
        # 导入图片
        import_action = QAction(QIcon(), "导入图片", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self._import_images)
        toolbar.addAction(import_action)
        
        toolbar.addSeparator()
        
        # 自动检测水印
        auto_detect_action = QAction(QIcon(), "自动检测水印", self)
        auto_detect_action.setShortcut("Ctrl+D")
        auto_detect_action.triggered.connect(self._auto_detect_watermark)
        toolbar.addAction(auto_detect_action)
        
        toolbar.addSeparator()
        
        # AI修复
        inpaint_action = QAction(QIcon(), "AI修复", self)
        inpaint_action.setShortcut("Ctrl+R")
        inpaint_action.triggered.connect(self._run_inpainting)
        toolbar.addAction(inpaint_action)
        
        toolbar.addSeparator()
        
        # 批量处理
        batch_action = QAction(QIcon(), "批量处理", self)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self._run_batch_processing)
        toolbar.addAction(batch_action)
        
        toolbar.addSeparator()
        
        # 保存结果
        save_action = QAction(QIcon(), "保存结果", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_result)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 查看Mask
        view_mask_action = QAction(QIcon(), "查看Mask", self)
        view_mask_action.setShortcut("Ctrl+M")
        view_mask_action.triggered.connect(self._view_mask)
        toolbar.addAction(view_mask_action)
        
        toolbar.addSeparator()
        
        # 下一张
        next_action = QAction(QIcon(), "下一张", self)
        next_action.setShortcut("Ctrl+N")
        next_action.triggered.connect(self._load_next_image)
        toolbar.addAction(next_action)
        
        # 上一张
        prev_action = QAction(QIcon(), "上一张", self)
        prev_action.setShortcut("Ctrl+P")
        prev_action.triggered.connect(self._load_prev_image)
        toolbar.addAction(prev_action)
    
    def _init_status_bar(self):
        """
        初始化状态栏
        """
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
    
    def _init_log_panel(self):
        """
        初始化底部日志面板
        """
        log_dock = QDockWidget("操作日志")
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_dock.setWidget(self.log_text)
    
    def _init_shortcuts(self):
        """
        初始化快捷键
        """
        # 已在工具栏中设置
    
    def _import_images(self):
        """
        导入图片（支持单张、多张、文件夹）
        """
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("图片文件 (*.jpg *.jpeg *.png *.webp)")
        
        if dialog.exec_():
            file_paths = [Path(p) for p in dialog.selectedFiles()]
            
            if file_paths:
                self.image_queue = file_paths
                self.current_queue_index = 0
                self._load_image(file_paths[0])
                self._log(f"已导入 {len(file_paths)} 张图片")
    
    def _load_image(self, file_path: Path):
        """
        加载图片
        """
        try:
            self.current_image_path = file_path
            self.current_image = ImageUtils.load_image(str(file_path))
            
            if self.current_image is not None:
                self.original_viewer.set_image(self.current_image)
                self.result_viewer.clear()
                self.original_viewer.clear_mask()
                self.current_mask = None
                self.result_image = None
                
                # 更新状态栏
                self.status_label.setText(f"当前图片: {file_path.name}")
                self._log(f"已加载图片: {file_path.name}")
        except Exception as e:
            logger.error(f"加载图片失败: {str(e)}")
            self._log(f"加载图片失败: {str(e)}")
    
    def _load_next_image(self):
        """
        加载下一张图片
        """
        if self.image_queue and self.current_queue_index < len(self.image_queue) - 1:
            self.current_queue_index += 1
            self._load_image(self.image_queue[self.current_queue_index])
    
    def _load_prev_image(self):
        """
        加载上一张图片
        """
        if self.image_queue and self.current_queue_index > 0:
            self.current_queue_index -= 1
            self._load_image(self.image_queue[self.current_queue_index])
    
    def _update_brush_size(self, size: int):
        """
        更新画笔大小
        """
        self.original_viewer.set_brush_size(size)
    
    def _clear_mask(self):
        """
        清空Mask
        """
        self.original_viewer.clear_mask()
        self.current_mask = None
        self._log("已清空Mask")
    
    def _on_mask_updated(self, mask):
        """
        Mask更新回调
        """
        self.current_mask = mask
        self._log("Mask已更新")
    
    def _auto_detect_watermark(self):
        """
        自动检测水印并生成Mask
        """
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先导入图片")
            return
        
        self.status_label.setText("正在自动检测水印...")
        
        # 在后台线程中执行检测
        class DetectThread(QThread):
            result_signal = pyqtSignal(object)
            
            def __init__(self, image, parent=None):
                super().__init__(parent)
                self.image = image
            
            def run(self):
                mask = self.parent().mask_generator.detect_watermark(self.image)
                self.result_signal.emit(mask)
        
        self.detect_thread = DetectThread(self.current_image, self)
        self.detect_thread.result_signal.connect(self._on_auto_detect_complete)
        self.detect_thread.start()
    
    def _on_auto_detect_complete(self, mask):
        """
        自动检测完成回调
        """
        if mask is not None:
            self.original_viewer.set_mask(mask)
            self.current_mask = mask
            self._log("自动检测完成，已生成Mask")
        else:
            self._log("未检测到水印区域")
        
        self.status_label.setText("就绪")
    
    def _run_inpainting(self):
        """
        执行AI修复
        """
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先导入图片")
            return
        
        if self.current_mask is None or self.current_mask.sum() == 0:
            QMessageBox.warning(self, "警告", "请先绘制或自动生成Mask")
            return
        
        self.status_label.setText("正在AI修复中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        # 在后台线程中执行修复
        class InpaintThread(QThread):
            result_signal = pyqtSignal(object)
            
            def __init__(self, image, mask, parent=None):
                super().__init__(parent)
                self.image = image
                self.mask = mask
            
            def run(self):
                try:
                    result = self.parent().lama_inpainter.inpaint(self.image, self.mask)
                    self.result_signal.emit(result)
                except Exception as e:
                    logger.error(f"修复失败: {str(e)}")
                    self.result_signal.emit(None)
        
        self.inpaint_thread = InpaintThread(self.current_image, self.current_mask, self)
        self.inpaint_thread.result_signal.connect(self._on_inpaint_complete)
        self.inpaint_thread.start()
    
    def _on_inpaint_complete(self, result):
        """
        修复完成回调
        """
        self.progress_bar.setVisible(False)
        
        if result is not None:
            self.result_image = result
            self.result_viewer.set_image(result)
            self._log("AI修复完成")
        else:
            QMessageBox.error(self, "错误", "AI修复失败，请查看日志")
        
        self.status_label.setText("就绪")
    
    def _run_batch_processing(self):
        """
        执行批量处理
        """
        if not self.image_queue:
            QMessageBox.warning(self, "警告", "请先导入图片")
            return
        
        # 选择输出目录
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        self.status_label.setText("批量处理中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.image_queue))
        
        # 使用自动检测模式批量处理
        self.batch_processor.process_batch(
            image_paths=self.image_queue,
            output_dir=output_dir,
            auto_detect=True
        )
    
    def _update_progress(self, current: int, total: int):
        """
        更新进度条
        """
        self.progress_bar.setValue(current)
        self.status_label.setText(f"处理进度: {current}/{total}")
    
    def _on_batch_complete(self, success_count: int, fail_count: int):
        """
        批量处理完成回调
        """
        self.progress_bar.setVisible(False)
        self.status_label.setText("批量处理完成")
        self._log(f"批量处理完成: 成功 {success_count} 张, 失败 {fail_count} 张")
        
        QMessageBox.information(
            self,
            "批量处理完成",
            f"成功: {success_count} 张\n失败: {fail_count} 张"
        )
    
    def _on_batch_error(self, message: str):
        """
        批量处理错误回调
        """
        self._log(f"批量处理错误: {message}")
    
    def _toggle_compare_mode(self):
        """
        切换对比模式
        """
        if self.current_image is not None and self.result_image is not None:
            # 创建对比视图
            compare_viewer = ImageViewer(enable_drawing=False)
            compare_viewer.set_compare_images(self.current_image, self.result_image)
            compare_viewer.show()
    
    def _view_mask(self):
        """
        查看Mask
        """
        if self.current_mask is None:
            QMessageBox.warning(self, "警告", "没有可用的Mask")
            return
        
        # 创建Mask查看窗口
        mask_window = QWidget()
        mask_window.setWindowTitle("Mask查看")
        mask_window.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(mask_window)
        
        # 创建标签说明
        label = QLabel("红色区域表示检测到的水印位置")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # 创建图片查看器显示Mask
        mask_viewer = ImageViewer(enable_drawing=False)
        
        # 将mask转换为可视化图像
        # 创建一个RGB图像，红色显示mask区域
        mask_visual = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 3), dtype=np.uint8)
        mask_visual[self.current_mask > 0] = [0, 0, 255]  # 红色
        
        mask_viewer.set_image(mask_visual)
        layout.addWidget(mask_viewer)
        
        # 添加统计信息
        total_pixels = self.current_mask.shape[0] * self.current_mask.shape[1]
        mask_pixels = np.sum(self.current_mask > 0)
        percentage = (mask_pixels / total_pixels) * 100
        
        stats_label = QLabel(f"Mask覆盖区域: {mask_pixels} 像素 ({percentage:.2f}%)")
        stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(stats_label)
        
        mask_window.show()
        self._log("已打开Mask查看窗口")
    
    def _save_result(self):
        """
        保存修复结果
        """
        if self.result_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return
        
        if self.current_image_path is None:
            default_name = "result.png"
        else:
            default_name = f"{self.current_image_path.stem}_clean.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            default_name,
            "PNG图片 (*.png);;JPG图片 (*.jpg)"
        )
        
        if file_path:
            try:
                ImageUtils.save_image(self.result_image, file_path)
                self._log(f"结果已保存: {file_path}")
                QMessageBox.information(self, "成功", "图片保存成功")
            except Exception as e:
                logger.error(f"保存失败: {str(e)}")
                QMessageBox.error(self, "错误", f"保存失败: {str(e)}")
    
    def _log(self, message: str):
        """
        添加日志信息
        """
        self.log_text.append(message)
        logger.info(message)
    
    def closeEvent(self, event):
        """
        关闭窗口事件
        """
        # 清理资源
        if hasattr(self, 'lama_inpainter'):
            self.lama_inpainter.cleanup()
        event.accept()
