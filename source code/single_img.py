import streamlit as st
import numpy as np
import cv2
import tempfile
from cnocr import CnOcr
from PIL import ImageFont, Image, ImageDraw

# 初始化 OCR 模型
ocr_model = CnOcr()

def get_image_path(image_input, input_type='local'):
    if input_type == 'local':
        return image_input
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_input.read())
            return tmp_file.name

def process_and_annotate_image(image_path):
    try:
        # 使用 OCR 模型识别文字
        results = ocr_model.ocr(image_path)
        recognized_texts = [item['text'] for item in results if item['score'] > 0. and not item['text'].isdigit() and item['text'] != '']
        result_text = ''.join(recognized_texts)

        # 使用 cv2 读取图像
        image = cv2.imread(image_path)
        
        # 提取边界框
        bboxes = [item['position'] for item in results if 'position' in item and item['score'] > 0. and not item['text'].isdigit() and item['text'] != '']

        # 设置字体 (请确保字体文件路径正确)
        font_path = "/Users/jolene/Downloads/SourceHanSerifTC-SemiBold.otf"
        font = ImageFont.truetype(font_path, 24)

        # 使用 cv2 和 PIL 混合进行绘图
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        centers = []
        for bbox in bboxes:
            cx = int(np.mean(bbox[:, 0]))
            cy = int(np.mean(bbox[:, 1]))
            centers.append((cx, cy))

            # 获取四个顶点的坐标
            points = bbox.astype(int)
            # 绘制多边形框
            cv2.polylines(image_rgb, [points], isClosed=True, color=(255, 0, 0), thickness=2)

        pil_img = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_img)

        for i in range(1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1]))
            print(f"距离框 {i-1} 和框 {i} 的距离：{dist:.2f}")

        # 转回 RGB 并保存结果
        annotated_image = np.array(pil_img)
        return annotated_image, result_text

    except Exception as e:
        st.error(f"图像处理失败: {str(e)}")
        return None, None

# Streamlit 界面
st.title("改进的单张图片处理与文字识别")

input_option = st.sidebar.selectbox("选择图像输入方式", ("本地路径", "上传文件"))

if input_option == "本地路径":
    image_path = st.text_input("输入本地图像路径")
    if st.button("加载图像") and image_path:
        annotated_image, recognized_text = process_and_annotate_image(image_path)
        if annotated_image is not None:
            st.image(annotated_image, channels="BGR", caption="处理后的图像", use_column_width=True)
            st.write(f"识别的文字: {recognized_text}")

elif input_option == "上传文件":
    uploaded_image = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        tmp_image_path = get_image_path(uploaded_image, input_type='upload')
        annotated_image, recognized_text = process_and_annotate_image(tmp_image_path)
        if annotated_image is not None:
            st.image(annotated_image, channels="BGR", caption="处理后的图像", use_column_width=True)
            st.write(f"识别的文字: {recognized_text}")
