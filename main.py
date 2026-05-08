
import sys
import os
import logging
from pathlib import Path

# 设置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watermark_remover.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication


def main():
    """
    主函数，启动应用程序
    """
    try:
        # 创建Qt应用
        app = QApplication(sys.argv)
        
        # 设置应用样式
        app.setStyle('Fusion')
        
        # 创建主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
