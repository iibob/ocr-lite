# 轻量版 OCR 文本识别

使用 ONNX Runtime 推理的轻量级 OCR，具备内存占用低、识别速度快的优势。

## 项目结构

```
ocr-lite/
├── models/
│   ├── angle_net.onnx                          # 方向检测模型
│   ├── crnn_lite.onnx                          # 文本识别模型
│   ├── db_net.onnx                             # 文本检测模型
│   └── keys.py                                 # 字符映射表
├── models_RapidOCR/
│   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx     # 方向检测模型
│   ├── ch_PP-OCRv4_det_infer.onnx              # PP-OCRv4 文本检测模型
│   ├── ch_PP-OCRv4_rec_infer.onnx              # PP-OCRv4 文本识别模型
│   ├── ch_PP-OCRv5_mobile_det.onnx             # PP-OCRv5 文本检测模型
│   ├── ch_PP-OCRv5_rec_mobile_infer.onnx       # PP-OCRv5 文本识别模型
│   ├── ppocr_keys_v1.txt                       # PP-OCRv4 字符映射表
│   └── ppocrv5_dict.txt                        # PP-OCRv5 字符映射表
├── OCR.py                                      # OCR 实现
├── rapidocr_text_rec.py                        # 仅用 RapidOCR 的识别模型进行文本识别
└── test_ocr.py                                 # 测试脚本

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

## test_ocr.py
测试脚本
```python
# 基础识别
test_ocr(ocr, img)

# 同时获取结果和可视化图像
test_ocr_visualization(ocr, img, output_path)

# 仅使用 CRNN 识别文本（需规范传入的图像，不然影响识别质量）
crnn_text_recognition_only()
```

## rapidocr_text_rec.py
- 仅调用 RapidOCR 文本识别模型推理
- 不做文本检测与方向判断
- 传入**已裁剪的单行文本区域的图片**
- 裁剪范围需略大于实际文本范围，不宜过大
- 可自行替换其他版本的 RapidOCR 模型

## ONNX Runtime 各版本下不同模型性能对比

| ONNX Runtime | v5 CPU % | 内存 MB | 耗时 s | v4 CPU % | 内存 MB | 耗时 s | lite CPU % | 内存 MB | 耗时 s |
| -------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| v1.23.2 | 1.8 | 72 | 0.008 | 1.8 | 60 | 0.008 | 1.6 | 55 | 0.007 |
| v1.23.1 | 2 | 73 | 0.008 | 1.9 | 61 | 0.008 | 1.6 | 55 | 0.007 |
| v1.23.0 | 1.8 | 73 | 0.008 | 1.8 | 61 | 0.008 | 1.6 | 55 | 0.008 |
| v1.22.1 | 1.7 | 78 | 0.008 | 1.8 | 62 | 0.008 | 1.7 | 55 | 0.007 |
| v1.22.0 | 1.8 | 81 | 0.008 | 1.9 | 61 | 0.008 | 1.7 | 57 | 0.007 |
| v1.21.1 | 1.9 | 81 | 0.008 | 1.7 | 62 | 0.008 | 1.7 | 55 | 0.007 |
| v1.21.0 | 1.8 | 79 | 0.008 | 1.7 | 59 | 0.008 | 1.7 | 57 | 0.007 |
| v1.20.1 | 1.7 | 70 | 0.008 | 1.7 | 51 | 0.008 | 1.7 | 45 | 0.007 |
| v1.20.0 | 1.7 | 73 | 0.008 | 1.8 | 52 | 0.008 | 1.8 | 50 | 0.007 |
| v1.19.2 | 1.7 | 73 | 0.008 | 1.7 | 52 | 0.008 | 1.7 | 50 | 0.007 |
| v1.19.0 | 1.6 | 70 | 0.008 | 1.8 | 53 | 0.008 | 1.7 | 47 | 0.007 |
| v1.18.1 | 1.9 | 74 | 0.008 | 1.8 | 52 | 0.008 | 1.7 | 49 | 0.007 |
| v1.18.0 | 1.8 | 74 | 0.008 | 1.8 | 52 | 0.008 | 1.7 | 47 | 0.007 |
| v1.17.3 | - | - | - | 1.7 | 52 | 0.008 | 1.8 | 49 | 0.007 |
| v1.17.1 | - | - | - | 1.8 | 53 | 0.008 | 1.6 | 46 | 0.007 |
| v1.17.0 | - | - | - | 1.8 | 54 | 0.008 | 1.7 | 48 | 0.007 |
| v1.16.3 | - | - | - | 1.8 | 52 | 0.008 | 1.5 | 46 | 0.007 |

### 说明
- v5：PP-OCRv5；v4：PP-OCRv4；lite：OCR-lite
- 本次仅基于文本识别模型，对同一张图片进行测试
- ONNX Runtime 版本小于 v1.18.0 时，不兼容 PP-OCRv5 模型
- 测试设备：
  - CPU型号：Intel(R) Core(TM) i9-10900 @ 2.80GHz（2.81 GHz） 
  - 内存容量：32 GB 
  - 操作系统：Windows 10 64位
- 补充说明：
  - 测试中，PP-OCRv5 模型的识别置信度相对表现更优
  - 同一模型下，不同配置的 CPU 仅影响识别耗时，不影响识别置信度


## 优化设想
- 批量推理提升识别效率：当前采用串行方式逐一推理识别图片文本，可将所有待识别图片统一处理尺寸后，批量送入模型推理，再完成批量解码，充分利用算力提升效率。

## License
- 本项目基于 [chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite) 进行二次开发，适配 Python3.10 并升级了依赖库，同时删除部分冗余库、精简冗余代码，还对部分逻辑做了优化调整，核心识别流程与实现逻辑沿用原仓库。模型来源：[查看](https://github.com/DayBreak-u/chineseocr_lite/tree/onnx/models)
- rapidocr_text_rec.py 参考 RapidOCR 开发。模型来源：[查看](https://www.modelscope.cn/models/RapidAI/RapidOCR/files)
