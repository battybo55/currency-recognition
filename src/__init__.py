"""纸币与硬币识别：预处理、分割、特征与分类模块。

模块规划（M3 起逐步落地）：
    io_utils   中文路径安全的读写、EXIF 处理
    preprocess 白平衡、去噪、光照校正、色彩空间
    segment    纸币四角定位与透视校正；硬币霍夫圆 / 分水岭
    features   颜色、LBP、GLCM、Gabor、几何、频域
    classify   规则先验 + SVM / RF / kNN / LR
    evaluate   指标计算、切片分析与结果落盘
"""

__version__ = "0.1.0"
