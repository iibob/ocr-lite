import os
import onnxruntime as ort
import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon
from models.keys import alphabet_chinese


class OCR:
    """OCR 总控"""
    def __init__(self, models_dir, angle_detect):
        ort.set_default_logger_severity(3)  # 屏蔽 WARNING/INFO 日志
        self.dbnet_max_size = 4096          # 输入图像长边的最大长度
        self.angle_detect = angle_detect    # 是否启用文字方向检测
        self.angle_detect_num = 30          # 参与方向判断的最大行数

        db_model_path = os.path.join(models_dir, "db_net.onnx")
        angle_net_path = os.path.join(models_dir, "angle_net.onnx")
        crnn_model_path = os.path.join(models_dir, "crnn_lite.onnx")

        self.detector = DBNet(db_model_path)
        self.recognizer = CRNN(crnn_model_path, alphabet_chinese)

        if self.angle_detect:
            self.angle_classifier = AngleNet(angle_net_path)

    def run(self, img, short_size=0, draw_box=False):
        """OCR主推理方法，依次执行文本检测、文字识别
        :param img: np.ndarray 输入图像（BGR格式）
        :param short_size: int，图像短边缩放尺寸，0=不压缩
        :param draw_box: bool，根据识别结果绘制检测框和序号
        :return: list，识别结果，每个元素为[文本框坐标, 带序号的识别文本, 检测置信度]
        """
        short_size = self.validate_and_align_short_size(short_size, img.shape)

        boxes, scores = self.detector.predict(img, short_size=short_size)
        results = self.recognize_text_from_boxes(img, boxes, scores)
        if not draw_box:
            return results

        vis_img = self.draw_results(img.copy(), results)
        return results, vis_img

    def validate_and_align_short_size(self, short_size, img_shape):
        """验证并对齐短边尺寸
        :param short_size: int，图像短边缩放尺寸，0=不压缩
        :param img_shape: tuple/list，原始图片的形状，格式为 (H, W, C)
        :return: int，验证+对齐后的合法短边尺寸，传入0则返回0
        """
        if short_size == 0:
            return 0

        try:
            short_size = int(short_size)
        except (ValueError, TypeError):
            raise ValueError("短边尺寸必须是整数，0=不压缩")

        if short_size < 64:
            raise ValueError("短边尺寸不能小于64px")

        short_size = 32 * (short_size // 32)  # 32倍数对齐

        # 计算图片等比例缩放后的长边尺寸是否超限
        img_h, img_w = img_shape[:2]
        scale = short_size / min(img_w, img_h)
        new_long_side = max(img_w, img_h) * scale

        if new_long_side > self.dbnet_max_size:
            raise ValueError(f"图片缩放后长边{new_long_side:.0f}px，超过最大限制{self.dbnet_max_size}px")

        return short_size

    def recognize_text_from_boxes(self, img, boxes_list, score_list):
        """对检测到的文本框逐一进行方向矫正并识别
        :param img: np.ndarray，原始输入图像（BGR格式）
        :param boxes_list: DBNet检测的文本框坐标列表 (N, 4, 2)
        :param score_list: 每个文本框对应的置信度
        :return: list，识别结果，每个元素为[文本框坐标, 带序号的识别文本, 检测置信度]
        """
        results = []
        boxes_list = self.sort_text_boxes(np.array(boxes_list))

        crops = []
        for box in boxes_list:
            crop = self.crop_rotate_text_box(img, box.astype(np.float32))
            crops.append(crop)

        is_text_rotated = False
        if self.angle_detect and len(crops) > 0:
            is_text_rotated = self.angle_classifier.predict(crops[:self.angle_detect_num])

        for crop, box, score in zip(crops, boxes_list, score_list):
            if self.angle_detect and is_text_rotated:
                crop = np.rot90(crop, 2)  # 180°

            text = self.recognizer.predict(crop)
            if text.strip():
                results.append([box, text, score])

        return results

    def sort_text_boxes(self, dt_boxes):
        """对文本框按从上到下、从左到右排序
        :param dt_boxes: np.ndarray，形状为[N,4,2]，N个文本框的坐标数组
        :return: 排序后的文本框列表
        """
        if len(dt_boxes) == 0:
            return []

        boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        y_thresh = np.mean([np.linalg.norm(b[0] - b[3]) for b in boxes]) * 0.5

        for i in range(len(boxes) - 1):
            y_diff = abs(boxes[i + 1][0][1] - boxes[i][0][1])
            if y_diff < y_thresh and boxes[i + 1][0][0] < boxes[i][0][0]:
                boxes[i], boxes[i + 1] = boxes[i + 1], boxes[i]

        return boxes

    def crop_rotate_text_box(self, img, points):
        """根据文本框四点坐标裁剪并进行透视矫正
        :param img: np.ndarray，原始输入图像（BGR格式）
        :param points: 文本框四点坐标 ndarray (4, 2)
        :return: np.ndarray，裁剪并矫正后的文本区域图像（BGR格式），异常时返回1×1×3的零矩阵
        """
        h, w = img.shape[:2]

        left = max(0, int(np.min(points[:, 0])))
        right = min(w, int(np.max(points[:, 0])))
        top = max(0, int(np.min(points[:, 1])))
        bottom = min(h, int(np.max(points[:, 1])))

        if right <= left or bottom <= top:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        img_crop = img[top:bottom, left:right]

        points[:, 0] -= left
        points[:, 1] -= top

        crop_w = int(np.linalg.norm(points[0] - points[1]))
        crop_h = int(np.linalg.norm(points[0] - points[3]))

        if crop_w <= 0 or crop_h <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        standard_points = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])

        transform_matrix = cv2.getPerspectiveTransform(points.astype(np.float32), standard_points)
        dst_img = cv2.warpPerspective(img_crop, transform_matrix, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE)

        if dst_img.shape[0] / max(1, dst_img.shape[1]) >= 1.5:
            dst_img = np.rot90(dst_img)

        return dst_img

    def draw_results(self, img, results):
        """绘制检测框和序号
        :param img: np.ndarray 输入图像
        :param results: OCR 结果
        :return: 绘制后的图像
        """
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)]

        for i, (box, _, _) in enumerate(results):
            pts = box.astype(int)
            color = colors[i % len(colors)]
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            x, y = pts[0]
            cv2.putText(img, str(i + 1), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1, lineType=cv2.LINE_AA)

        return img


class DBNet:
    """文本检测模型推理"""
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)
        self.min_size = 3            # 文本框最小边长阈值，小于该值过滤
        self.thresh = 0.3            # 概率图二值化阈值
        self.score_thresh = 0.5      # 文本框得分阈值，小于该值过滤
        self.max_candidates = 1000   # 最大候选框数量，防止轮廓过多
        self.unclip_ratio = 2.0      # 文本框膨胀系数，扩大检测框以覆盖完整文本

        self.mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)   # 图像归一化均值
        self.std = np.array((0.229, 0.224, 0.225), dtype=np.float32)    # 图像归一化标准差

    def predict(self, img, short_size):
        """对单张图片执行文本检测
        :param img: np.ndarray，输入图像
        :param short_size: int，图像短边缩放尺寸（0=不缩放，用原图尺寸；>0按短边缩放且为32的倍数）
        :return: tuple，(boxes, scores)，boxes为[N,4,2]的文本框坐标数组，scores为[N,]的置信度数组
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        if short_size == 0:
            new_h, new_w = orig_h, orig_w
        else:
            if orig_h < orig_w:
                scale = short_size / orig_h
                new_h = short_size
                new_w = int(orig_w * scale)
            else:
                scale = short_size / orig_w
                new_w = short_size
                new_h = int(orig_h * scale)

        new_h = max(32, new_h - new_h % 32)
        new_w = max(32, new_w - new_w % 32)

        img = cv2.resize(img, (new_w, new_h))

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        probability_map = self.sess.run(["out1"], {"input0": img})[0][0]

        boxes, scores = self.decode(probability_map, orig_h, orig_w)
        return boxes, scores

    def decode(self, pred, dest_h, dest_w):
        """对模型输出的概率图进行解码，提取轮廓、生成文本框、过滤低质量框
        :param pred: 模型输出概率图
        :param dest_h: 原图高度
        :param dest_w: 原图宽度
        :return: tuple，(boxes, scores)，过滤后的文本框坐标数组和对应置信度数组
        """
        probability_map = pred[0]
        bitmap = probability_map > self.thresh

        h, w = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        scores = []

        for contour in contours[:self.max_candidates]:
            contour = contour.squeeze(1)

            if cv2.contourArea(contour) < self.min_size * self.min_size:
                continue

            box, min_side = self.get_minimum_bounding_box(contour)
            if min_side < self.min_size:
                continue

            score = self.calculate_box_score(probability_map, contour)
            if score < self.score_thresh:
                continue

            expanded = self.unclip(np.array(box), self.unclip_ratio)
            if expanded is None or len(expanded) == 0:
                continue

            expanded = expanded[0]

            box, min_side = self.get_minimum_bounding_box(expanded)
            if min_side < self.min_size + 2:
                continue

            box = np.array(box)

            # 映射回原图
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)

            boxes.append(box.astype(np.int16))
            scores.append(score)

        if not boxes:
            return np.zeros((0, 4, 2), dtype=np.int16), np.zeros((0,), dtype=np.float32)

        return np.array(boxes), np.array(scores, dtype=np.float32)

    def get_minimum_bounding_box(self, contour):
        """根据轮廓点生成最小外接矩形，对矩形顶点按左上、右上、右下、左下排序
        :param contour: np.ndarray，轮廓点坐标数组，形状为[M,2]
        :return: tuple，(box_points, min_side)，box_points为4个顶点的坐标列表，min_side为矩形最短边长
        """
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)
        points = sorted(points, key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            tl, bl = points[0], points[1]
        else:
            tl, bl = points[1], points[0]

        if points[3][1] > points[2][1]:
            tr, br = points[2], points[3]
        else:
            tr, br = points[3], points[2]

        return [tl, tr, br, bl], min(rect[1])

    def calculate_box_score(self, probability_map, box):
        """计算文本框内的平均置信度
        :param probability_map: np.ndarray，二值化前的概率图，形状为[H,W]
        :param box: np.ndarray，文本框的四个顶点坐标数组，形状为[4,2]
        :return: float，文本框的置信度值
        """
        h, w = probability_map.shape
        box = box.copy()

        xmin = np.clip(int(np.floor(box[:, 0].min())), 0, w - 1)
        xmax = np.clip(int(np.ceil(box[:, 0].max())), 0, w - 1)
        ymin = np.clip(int(np.floor(box[:, 1].min())), 0, h - 1)
        ymax = np.clip(int(np.ceil(box[:, 1].max())), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin

        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
        return cv2.mean(probability_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box, unclip_ratio=1.5):
        """对文本框进行膨胀处理，扩大框体以覆盖完整的文本区域
        :param box: np.ndarray，文本框的四个顶点坐标数组，形状为[4,2]
        :param unclip_ratio: float，膨胀系数
        :return: np.ndarray/[]，膨胀后的多边形顶点坐标，面积为0时返回空列表
        """
        poly = Polygon(box)
        if poly.area <= 0:
            return []

        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance))


class AngleNet:
    """文本角度（0° / 180°）检测模型推理"""
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)
        self.size_h = 32    # 网络输入图像高度
        self.size_w = 192   # 网络输入图像最大宽度

    def predict(self, imgs):
        """对多张文本图片进行方向投票判断
        :param imgs: list，文本行裁剪图像列表，每个元素为np.ndarray（BGR格式）
        :return: bool，True表示需要180°旋转矫正，False表示无需矫正
        """
        reversed_count = 0
        for img in imgs:
            reversed_count += self.predict_for_single_image(img)
        threshold = max(1, len(imgs) // 2)
        return reversed_count < threshold

    def predict_for_single_image(self, img):
        """对单张图片进行方向预测
        :param img: np.ndarray，文本行裁剪图像（BGR格式）
        :return: int，预测类别（0=正向，1=倒置）
        """
        h, w = img.shape[:2]
        scale = h / self.size_h
        new_w = max(1, int(w / scale))

        img = cv2.resize(img, (new_w, self.size_h), interpolation=cv2.INTER_LINEAR)

        if new_w < self.size_w:
            padded = np.full((self.size_h, self.size_w, 3), 255, dtype=np.uint8)
            padded[:, :new_w] = img
            img = padded
        else:
            img = img[:, :self.size_w]

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]

        logits = self.sess.run(["out"], {"input": img})[0]
        return int(np.argmax(logits, axis=1)[0])


class CRNN:
    """文本识别模型推理"""
    def __init__(self, model_path, alphabet):
        self.sess = ort.InferenceSession(model_path)
        self.alphabet = alphabet    # 字符映射表

    def predict(self, img):
        """文本识别
        :param img: np.ndarray，文本行裁剪矫正后的图像（BGR格式）
        :return: str，识别出的字符串
        """
        h, w = img.shape[:2]
        scale = h / 32
        new_w = max(1, int(w / scale))

        img = cv2.resize(img, (new_w, 32))

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]

        logits = self.sess.run(["out"], {"input": img})[0]
        return self.decode(logits)

    def decode(self, logits):
        """把模型输出的数字标签序列还原成原始字符串
        :param logits: np.ndarray，模型输出的预测值，形状为[W, C]（W为序列长度，C为字符类别数）
        :return: str，解码后的中文字符串
        """
        if logits.ndim == 3:
            logits = logits[:, 0, :]

        preds = np.argmax(logits, axis=1)

        char_list = []
        prev_idx = -1

        for idx in preds:
            if idx != 0 and idx != prev_idx:
                char_list.append(self.alphabet[idx - 1])
            prev_idx = idx

        return ''.join(char_list)
