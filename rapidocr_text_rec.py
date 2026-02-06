import cv2
import numpy as np
import onnxruntime as ort
import math
import time
import os


class OCRRecognizer:
    """OCR 识别器，仅用识别模型，识别已裁剪图片，不检测文本位置、判断方向"""
    def __init__(self, rec_model_path, rec_keys_path, use_model_v5=False):
        self.rec_keys_path = rec_keys_path      # 字符字典文件路径 (.txt 文件)
        self.rec_img_shape = (3, 48, 320)       # 识别模型输入图像形状 (C, H, W)
        self.use_model_v5 = use_model_v5        # 使用 PP-OCRv5 模型

        self.character = self.load_character_dict()             # 加载字符字典
        self.session = ort.InferenceSession(rec_model_path)     # 初始化 ONNX Runtime 会话

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = [self.session.get_outputs()[0].name]

    def predict(self, img):
        """识别图像中的文字
        :param img: numpy 数组图像 (H, W, C)
        :return: 识别的文本和置信度
        """
        # 转换为 RGB
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 预处理
        norm_img = self.resize_norm_img(img)

        # 推理
        outputs = self.session.run(self.output_name, {self.input_name: norm_img})
        preds = outputs[0]

        # 后处理
        preds_idx = preds.argmax(axis=2)[0]
        preds_prob = preds.max(axis=2)[0]

        # 组合索引和置信度
        text_index = list(zip(preds_idx, preds_prob))

        # 解码
        text, confidence = self.decode_text(text_index)
        return text, confidence

    def load_character_dict(self):
        """加载字符字典"""
        with open(self.rec_keys_path, 'r', encoding='utf-8') as f:
            dict_characters = [line.strip() for line in f if line.strip()]

        if self.use_model_v5:
            character = ['blank', 'blank'] + dict_characters
        else:
            character = ['blank'] + dict_characters

        return character

    def resize_norm_img(self, img):
        """对图像进行 resize 和归一化处理
        :param img: img: 输入图像 (H, W, C)
        :return: 处理后的图像 (1, C, H, W)
        """
        imgC, imgH, imgW = self.rec_img_shape

        # 计算缩放比例，保持宽高比
        h, w = img.shape[:2]
        ratio = w / float(h)

        # 根据高度固定为 imgH，计算宽度
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        # Resize 图像
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')

        # 归一化
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0  # HWC -> CHW
        resized_image -= 0.5
        resized_image /= 0.5

        # Padding 到固定宽度
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image

        # 增加 batch 维度
        return padding_im[np.newaxis, :]

    def decode_text(self, text_index):
        """将模型输出的索引解码为文本
        :param text_index: 模型输出的字符索引
        :return: 识别的文本和置信度
        """
        result_list = []
        conf_list = []
        prev_char_idx = None

        for idx, conf in text_index:
            idx = int(idx)
            if idx == 0:
                prev_char_idx = None  # 重置分段起点
                continue    # 跳过 blank

            if idx == prev_char_idx:
                continue    # 跳过同一段内的连续重复字符

            if idx < len(self.character):
                result_list.append(self.character[idx])
                conf_list.append(conf)  # 添加字符
                prev_char_idx = idx     # 更新为当前字符

        text = ''.join(result_list)
        confidence = np.mean(conf_list) if conf_list else 0.0
        return text, float(confidence)


def main():
    use_model_v5 = True     # 使用 PP-OCRv5 模型推理
    # use_model_v5 = False    # 使用 PP-OCRv4 模型推理

    models_dir = "models_RapidOCR"

    if use_model_v5:
        rec_model_path = os.path.join(models_dir, "ch_PP-OCRv5_rec_mobile_infer.onnx")
        rec_keys_path = os.path.join(models_dir, "ppocrv5_dict.txt")
    else:
        rec_model_path = os.path.join(models_dir, "ch_PP-OCRv4_rec_infer.onnx")
        rec_keys_path = os.path.join(models_dir, "ppocr_keys_v1.txt")

    # 读取测试图片
    test_image = "test_image_2.png"
    img = cv2.imread(str(test_image))

    # 创建识别器
    recognizer = OCRRecognizer(
        rec_model_path=rec_model_path,
        rec_keys_path=rec_keys_path,
        use_model_v5=use_model_v5
    )

    print(f"\n正在识别图像: {test_image}")
    start = time.time()
    text, confidence = recognizer.predict(img)
    print(f"识别结果:")
    print(f"  文  本: {text}")
    print(f"  置信度: {confidence:.2%}")
    print(f"  耗  时：{time.time() - start:.8f} 秒")


if __name__ == "__main__":
    main()
