
import os
import subprocess
import sys

def build():
    """
    使用PyInstaller打包项目
    """
    # 检查PyInstaller是否安装
    try:
        import PyInstaller
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 构建命令
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=AI去水印工具",
        "--add-data=ui/*;ui/",
        "--add-data=core/*;core/",
        "--add-data=models/*;models/",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=lama_cleaner",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "main.py"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("打包完成！")
    print("输出文件位于 dist/ 目录")

if __name__ == "__main__":
    build()
