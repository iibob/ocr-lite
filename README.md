# 轻量版 OCR 文本识别

使用 ONNX Runtime 推理的轻量级 OCR，具备内存占用低、识别速度快的优势，同时文本误识别率控制在可接受范围。

## 项目结构

```
ocr-lite/
├── models/
│   ├── angle_net.onnx     # 方向检测模型
│   ├── crnn_lite.onnx     # 文本识别模型
│   ├── db_net.onnx        # 文本检测模型
│   └── keys.py            # 字符映射表
├── OCR.py                 # OCR 实现
└── test_ocr.py            # 测试脚本
```

## 快速开始

```python
from OCR import OCR
import cv2

# 初始化OCR引擎
ocr_engine = OCR(
  models_dir="models",  # 模型文件目录
  angle_detect=True     # 是否启用方向检测
)

# 读取图像
img = cv2.imread("test_image_1.png")

# 执行OCR识别
results = ocr_engine.run(img)

# 打印结果
for i, (box, text, score) in enumerate(results):
  print(f"[{i + 1}] {text} (置信度: {score:.2%})")
  print(f"    位置: {box.tolist()}")

# 同时返回识别结果和标注图像
results, vis_img = ocr_engine.run(img, short_size=960, draw_box=True)

# 保存可视化结果
cv2.imwrite("test_result.png", vis_img)
```

## 参数说明

### OCR 初始化参数

- `models_dir`: 模型文件所在目录
- `angle_detect`: 是否启用文字方向检测，默认 False

### run() 方法参数

- `img`: 输入图像（OpenCV BGR 格式）
- `short_size`: 图像短边目标尺寸（默认为 0，表示不缩放）
- `draw_box`: 是否保存可视化结果，默认 False

## 测试脚本说明
```python
# 基础识别
test_ocr(ocr, img)

# 同时获取结果和可视化图像
test_ocr_visualization(ocr, img, output_path)

# 仅使用 CRNN 识别文本（需规范传入的图像，不然影响识别质量）
crnn_text_recognition_only()
```

## 性能优化设想
- 批量推理提升识别效率：当前采用串行方式逐一推理识别图片文本，可将所有待识别图片统一处理尺寸后，批量送入模型推理，再完成批量解码，充分利用算力提升效率。
- 简单场景简化识别流程：针对纯色背景、单行水平文字的场景，通过 cv2 提取文字区域后，直接送入 CRNN 执行文本识别，无需运行 DBNet 检测、AngleNet 角度检测环节，大幅节省耗时。

## 致谢
- 本项目基于 [chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite) 进行二次开发，适配 Python3.10 并升级了依赖库，同时删除部分冗余库、精简冗余代码，还对部分逻辑做了优化调整，核心识别流程与实现逻辑沿用原仓库。
- 模型来源：[查看](https://github.com/DayBreak-u/chineseocr_lite/tree/onnx/models)
