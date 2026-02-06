import cv2
from OCR import OCR
import time


def print_results(results, cost_time):
    print(f"共识别到 {len(results)} 个文本区域，耗时：{cost_time:.4f} 秒")
    print("-" * 60)
    for i, (box, text, score) in enumerate(results, 1):
        print(f"[{i}] {text}")
        print(f"    置信度: {score:.2%}")
        print(f"    位置: 左上({box[0][0]:.0f},{box[0][1]:.0f}) 右下({box[2][0]:.0f},{box[2][1]:.0f})")
        print()
    print("-" * 60)


def test_ocr(ocr, img):
    """OCR测试
    :param ocr: OCR引擎实例
    :param img: 输入图像
    """
    start_time = time.time()
    results = ocr.run(img)
    cost_time = time.time() - start_time

    if not results:
        print("未检测到任何文本")
        return

    print_results(results, cost_time)


def test_ocr_visualization(ocr, img, output_path):
    """带可视化的OCR测试
    :param ocr: OCR引擎实例
    :param img: 输入图像
    :param output_path: 可视化结果图片路径
    """
    start_time = time.time()
    results, vis_img = ocr.run(img, short_size=960, draw_box=True)
    cost_time = time.time() - start_time
    print_results(results, cost_time)

    cv2.imwrite(output_path, vis_img)
    print(f"可视化结果已保存到: {output_path}")


def crnn_text_recognition_only():
    """CRNN 轻量版文本识别（需规范传入的图像，不然影响识别质量）"""
    from OCR import CRNN
    from models.keys import alphabet_chinese

    image_path = "test_image_2.png"
    crnn_model_path = "models/crnn_lite.onnx"
    recognizer = CRNN(crnn_model_path, alphabet_chinese)

    img = cv2.imread(image_path)
    start_time = time.time()
    result = recognizer.predict(img)
    print(f"识别结果：{result}")
    print(f"识别耗时：{time.time() - start_time:.4f} 秒")


def main():
    # 配置路径
    models_dir = "models"               # 模型目录
    image_path = "test_image_1.png"     # 需要识别的图片
    output_path = "test_result.png"     # 生成可视化结果（检测框+序号）
    # angle_detect = True                 # 启用文字方向检测
    angle_detect = False

    # 初始化OCR引擎
    ocr = OCR(models_dir=models_dir, angle_detect=angle_detect)

    img = cv2.imread(image_path)
    print(f"测试图像尺寸: {img.shape[1]} x {img.shape[0]}")

    # 测试1: 基础识别
    test_ocr(ocr, img)

    # 测试2: 可视化
    # test_ocr_visualization(ocr, img, output_path)


if __name__ == "__main__":
    main()
    # crnn_text_recognition_only()
