import streamlit as st
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.models import densenet121
from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models
import cv2
import plotly.express as px
import torch.nn as nn
from cnocr import CnOcr
import traceback
import re

# 設定中文字
font_path = "SourceHanSerifTC-SemiBold.otf"  
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()

# 加載訓練好的模型
@st.cache_data()
def load_model():
    model = densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(in_features=1024, out_features=5)
    state_dict = torch.load('best_model_DenseNet.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_model_ca():
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    num_classes = 15
    vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes) 
    state_dict = torch.load('best_model_CL.pth', map_location=torch.device('cpu'))
    vit.load_state_dict(state_dict, strict=False)
    vit.heads.head = torch.nn.Identity()
    vit.eval()
    return vit

# 文字辨識
def load_ocr():
    return CnOcr()

#import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# 自定義標準化類別
class My_normalization(object):
    def __call__(self, image):
        # 檢查圖像是否已經是Tensor
        if not isinstance(image, torch.Tensor):
            image_tensor = transforms.ToTensor()(image)  # 轉為 Tensor
        else:
            image_tensor = image

        new_image = (image_tensor - image_tensor.mean()) / (image_tensor.max() - image_tensor.min())
        return new_image

transform_vit = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image, model_type):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    if model_type == 'densenet':
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(gray_img, (224, 224))
        retval, binary_img = cv2.threshold(resized_img, 240, 255, cv2.THRESH_OTSU)
        processed_img = 255 - binary_img  # 反轉

        # 轉換回 3 通道圖像
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

        # 使用 torchvision 的 transforms 進行數據增強和標準化
        transform = transforms.Compose([
            transforms.ToTensor(),
            My_normalization()  # 自定義的標準化
        ])

        img_tensor = transform(processed_img)
        img_tensor = img_tensor.unsqueeze(0)  # 增加批量維度
        
    elif model_type == 'vit':
        img_tensor = transform_vit(img).unsqueeze(0)
        
    elif model_type == 'ocr':
        img_tensor = img
    
    return img_tensor

# 預測字體分類
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 文字辨識
def process_and_annotate_image(image):
    
    result = ocr_model.ocr(image)
    recognized_texts = [item['text'] for item in result if item['score'] > 0.1 and not item['text'].isdigit() and item['text'] != ''  and not re.search(r'[A-Za-z]', item['text'])]
    result_text = ''.join(recognized_texts)

    bboxes = [item['position'] for item in result if 'position' in item if item['score'] > 0.1 and not item['text'].isdigit() and item['text'] != '']
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = process_image(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, 48)
    
    centers = []
    for i, (bbox, text) in enumerate(zip(bboxes, recognized_texts)):
        # 绘制多边形框
        points = bbox.astype(int)
        # cv2.polylines(image_rgb, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        polygon = [(p[0], p[1]) for p in points]  # 转换为 tuple list 以绘制多边形
        draw.line(polygon + [polygon[0]], fill=(0, 255, 0), width=5)  # 画线条

        # 绘制文字内容和编号
        x, y = points[0]  # 使用左上角作为文本位置
        draw.text((x, y - 25), f"{i+1} {text}", font=font, fill=(0, 255, 0))
        # 计算中心点
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))
        centers.append((cx, cy))

    word_dist = []
    total_distance = 0  # 初始化总距离
    num_distances = len(centers) - 1  # 距离的数量

    for i in range(1, len(centers)):
        dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1]))
        total_distance += dist  # 累加总距离
        word_dist.append(dist)
        
    if num_distances > 0:
        avg_dist = total_distance / num_distances
    else:
        avg_dist = 0
        
    # 转换回 OpenCV 格式
    annotated_image = np.array(pil_img)
    return annotated_image, result_text, avg_dist

def process_image(image):
    # 讀取影像並轉換為 HSV 顏色空間
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = np.array(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅色範圍
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 提取紅色部分的掩膜
    mask_red = cv2.inRange(hsv_image, lower_red1, upper_red1) | cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # 如果印章不是紅色，則提取深色區域（黑色）作為備選
    if cv2.countNonZero(mask_red) < 500:  # 假設小於500個像素為紅色
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask_red = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 使用膨脹和腐蝕來處理印章區域
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    
    # 使用掩膜提取印章並反轉成白底黑字
    result = cv2.bitwise_and(image, image, mask=mask_cleaned)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary_result = cv2.threshold(gray_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_result = cv2.bitwise_not(binary_result)
    
    # 轉換為 RGB 並返回 PIL Image 格式
    rgb_result = cv2.cvtColor(inverted_result, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(rgb_result)
    return pil_image


# 從 ZIP 文件中讀取所有圖像
def load_images_from_zip(zip_file):
    images = []
    image_names = []
    with zipfile.ZipFile(zip_file, 'r') as archive:
        for file in archive.namelist():
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_data = archive.read(file)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                    image_names.append(file)
    return images, image_names

# 提取特徵的函數
def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
    return features

# 建立 Streamlit 網頁界面
st.title('AIGO 潛力新星盃 - 詐騙文件印鑑、關防圖章 AI 辨識')

label_mapping = {0: '宋體', 1: '古印', 2: '方篆', 3: '楷書', 4: '楷書'}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
prediction_counts = {label: 0 for label in reverse_label_mapping.keys()}

# 上傳壓縮包（ZIP）
uploaded_zip = st.file_uploader("上傳一個包含圖片的壓縮包 (ZIP)", type=["zip"])

# 上傳 NPZ 文件
uploaded_npz = st.file_uploader("上傳 NPZ 文件", type=["npz"])

# 自體分類結果字典
self_classification_results = {}

if uploaded_zip is not None:
    # 加載模型
    model = load_model()
    vit_model = load_model_ca()
    ocr_model = load_ocr()

    # 從壓縮包中讀取圖像
    images, image_names = load_images_from_zip(uploaded_zip)
    image_names = [name.split('/')[-1] for name in image_names]

    # 加載 NPZ 文件
    if uploaded_npz is not None:
        npz_data = np.load(uploaded_npz)
        local_features = npz_data['features']  # 假設您在 NPZ 中使用 'features' 鍵
        local_labels = npz_data['labels']  # 假設您在 NPZ 中使用 'labels' 鍵
    else:
        local_features = None
        local_labels = None

    if images:
        images_per_row = 5

# 初始化分類統計字典
        classification_stats = {label: {'count': 0, 'max_similarity': 0} for label in local_labels}

# 處理圖像
        for i in range(0, len(images), images_per_row):  # 每 5 張圖像處理一排
            cols = st.columns(images_per_row)  # 創建 5 列

            for idx, img in enumerate(images[i:i + images_per_row]):  # 每排最多顯示 5 張
                try:
            # 預處理圖像
                    image_tensor = preprocess_image(img, model_type='densenet')

            # 預測字體類別
                    predicted_class = predict_image(model, image_tensor)
                    predicted_label = label_mapping[predicted_class]
                    prediction_counts[predicted_label] += 1
                    
                    annotated_image, result_text, avg_dist = process_and_annotate_image(img)

            # 顯示圖像和預測結果
                    cols[idx].image(annotated_image, caption=f'{image_names[i + idx]}: {predicted_label}', use_column_width=True)
                    # cols[idx].write(f"文字辨識結果: {result_text}")
                    cols[idx].write(f"字數: {len(result_text)}")
                    cols[idx].write(f'平均字距：{avg_dist:.2f}')

            # 自體分類功能（使用 ViT 特徵）
                    vit_image_tensor = preprocess_image(img, model_type='vit')
                    uploaded_feature = extract_features(vit_model, vit_image_tensor)
                    if local_features is not None and local_labels is not None:
                # 計算相似度
                        uploaded_feature = uploaded_feature.reshape(1, -1)
                        local_features = local_features.reshape(local_features.shape[0], -1)

                        similarities = cosine_similarity(uploaded_feature, local_features)
                        most_similar_index = np.argmax(similarities)
                        most_similar_label = local_labels[most_similar_index]
                        max_similarity = similarities[0][most_similar_index]

                # 更新統計
                        classification_stats[most_similar_label]['count'] += 1
                        classification_stats[most_similar_label]['max_similarity'] = max(max_similarity, classification_stats[most_similar_label]['max_similarity'])

                # 將相似度結果存儲到字典中
                        if most_similar_label not in self_classification_results:
                            self_classification_results[most_similar_label] = []
                        self_classification_results[most_similar_label].append((img, max_similarity))

                        cols[idx].write(f"詐騙來源分類: {most_similar_label}，相似度: {max_similarity:.4f}")
                        
                except Exception as e:
                    cols[idx].write(f"讀取圖像失敗: {e}")
                    st.write(traceback.format_exc())

# 顯示分類統計
    st.subheader("分類統計結果")
    for label, stats in classification_stats.items():
        st.write(f"**{label}**: 數量 = {stats['count']}, 最大相似度 = {stats['max_similarity']:.4f}")

    # 顯示預測結果統計
    st.subheader('\n預測結果統計')
    total_predictions = sum(prediction_counts.values())  # 總預測數量
    labels = list(prediction_counts.keys())
    counts = list(prediction_counts.values())

    percentages = np.array(counts) / total_predictions * 100

    new_labels = []
    new_counts = []
    other_count = 0

    # 將小於5%的類別合併為 "其他"
    for i, percent in enumerate(percentages):
        if percent < 5:
            other_count += counts[i]
        else:
            new_labels.append(labels[i])
            new_counts.append(counts[i])

    if other_count > 0:
        new_labels.append("其他")
        new_counts.append(other_count)

    # 統計圖表顯示
    # fig = px.bar(x=new_labels, y=new_counts, labels={'x': '字體類別', 'y': '預測次數'}, title='各字體類別預測次數')
    # st.plotly_chart(fig)

# 圓餅圖
    fig = px.pie(values=new_counts, names=new_labels, title='各字體類別預測比例', 
             color=new_labels, 
             color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

        # 顯示分類統計結果
    st.subheader("各分類的統計資料")
    labels = list(classification_stats.keys())
    counts = [stat['count'] for stat in classification_stats.values()]
    max_similarities = [stat['max_similarity'] for stat in classification_stats.values()]

# 條形圖視覺化
    fig = px.bar(x=labels, y=counts, labels={'x': '來源類別', 'y': '預測次數'}, title='各來源類別預測次數')
    st.plotly_chart(fig)
    
    # fig = px.pie(values=counts, names=labels, title='各來源類別預測比例', 
    #          color=labels, 
    #          color_discrete_sequence=px.colors.qualitative.Plotly)
    # st.plotly_chart(fig)