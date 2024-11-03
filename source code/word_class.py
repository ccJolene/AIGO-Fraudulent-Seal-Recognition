import streamlit as st
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision import models, transforms
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import zipfile
from io import BytesIO
from torchvision.models import densenet121
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

# 設定中文字
font_path = "/Users/jolene/Downloads/SourceHanSerifTC-SemiBold.otf"  
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()


# 加載訓練好的模型
@st.cache_data()
def load_model():
    model = densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(in_features=1024, out_features=8)
    state_dict = torch.load('best_model_DenseNet.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 圖像預處理
def preprocess_image(image):
    # 轉換為灰度並調整大小
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))

    # Otsu 閾值處理
    retval, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_OTSU)
    processed_img = 255 - binary_img  # 反轉

    # 轉換回 3 通道圖像
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

    # 使用 torchvision 的 transforms 進行數據增強和標準化
    transform = transforms.Compose([
        transforms.ToTensor(),               # 轉為Tensor
        My_normalization()                 # 自定義的標準化 # 標準化
    ])

    img_tensor = transform(processed_img)
    img_tensor = img_tensor.unsqueeze(0)  # 增加批量維度
    return img_tensor

# 自定義標準化類別
class My_normalization(object):
    def __call__(self, image):
        # 檢查圖像是否已經是Tensor
        if not isinstance(image, torch.Tensor):
            # 如果不是Tensor，先轉為Tensor
            image_tensor = F.to_tensor(image)
        else:
            image_tensor = image  # 如果已經是Tensor，直接使用
        
        # 將Tensor轉換為浮點數型
        image_tensor = F.convert_image_dtype(image_tensor, dtype=torch.float)
        # 自定義標準化公式: (image - mean) / (max - min)
        new_image = (image_tensor - image_tensor.mean()) / (image_tensor.max() - image_tensor.min())
        return new_image

# 預測字體分類
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 遞迴遍歷資料夾，找到所有圖片檔案
def find_images_in_folder(folder):
    image_files = []
    # 遞迴遍歷資料夾及其子資料夾
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):  # 檔案格式過濾
                image_files.append(os.path.join(root, file))
    return image_files

# 從 ZIP 文件中讀取所有圖像，包含子資料夾
def load_images_from_zip(zip_file):
    images = []
    image_names = []
    with zipfile.ZipFile(zip_file, 'r') as archive:
        for file in archive.namelist():
            # 檢查文件是否為圖像 (忽略大小寫)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_data = archive.read(file)  # 讀取圖像數據
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)  # 解碼圖像
                if img is not None:
                    images.append(img)  # 添加圖像
                    image_names.append(file)  # 記錄圖像名稱
    return images, image_names

# 建立 Streamlit 網頁界面
st.title('AIGO 潛力新星盃 - 詐騙文件印鑑、關防圖章 AI 辨識')

label_mapping = {'宋體': 0, '隸書':1, '方篆':2, '楷書':3, '小篆':4, '行書':5, '圓體':6, '印刷體':7}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
prediction_counts = {label: 0 for label in reverse_label_mapping.values()}

# 上傳壓縮包（ZIP）
uploaded_zip = st.file_uploader("上傳一個包含圖片的壓縮包 (ZIP)", type=["zip"])

if uploaded_zip is not None:
    # 加載模型
    model = load_model()

    # 從壓縮包中讀取圖像
    images, image_names = load_images_from_zip(uploaded_zip)
    image_names = [os.path.splitext(name)[0] for name in image_names]

    if images:
        for i in range(0, len(images), 10):  # 每 15 張圖像處理一排
            cols = st.columns(10)  # 創建 15 列

            for idx, img in enumerate(images[i:i+10]):  # 每排最多顯示 15 張
                try:
                    # 預處理圖像
                    image_tensor = preprocess_image(img)

                    # 預測字體類別
                    predicted_class = predict_image(model, image_tensor)
                    predicted_label = reverse_label_mapping[predicted_class]
                    prediction_counts[predicted_label] += 1

                    # 顯示圖像和分類結果
                    cols[idx].image(img, caption=f'{image_names[i+idx]}:{predicted_label}', use_column_width=True)
                except Exception as e:
                    cols[idx].write(f"讀取圖像失敗: {e}")
    else:
        st.write("壓縮包中沒有找到圖片文件")
        
        
    st.subheader('\n預測結果統計')
    total_predictions = sum(prediction_counts.values())  # 总预测数量
    labels = list(prediction_counts.keys())
    counts = list(prediction_counts.values())

    # labels = prediction_counts.keys()
    # counts = prediction_counts.values()
    percentages = np.array(counts) / total_predictions * 100

    new_labels = []
    new_counts = []
    other_count = 0

    for i, percent in enumerate(percentages):
        if percent < 5:
            other_count += counts[i]  # 合并小于5%的类别
        else:
            new_labels.append(labels[i])
            new_counts.append(counts[i])

    # 如果有 "其他" 类别，添加到最后
    if other_count > 0:
        new_labels.append("其他")
        new_counts.append(other_count)

    # 设置一致的颜色
    colors = plt.get_cmap('tab20').colors  # 使用一致的颜色方案
    color_mapping = colors[:len(new_labels)]
        
    fig, ax = plt.subplots(1, 2)
    ax[0].bar(new_labels, new_counts, color=color_mapping)
    ax[0].set_xlabel("字體類別", fontproperties=my_font)
    ax[0].set_ylabel("預測次數", fontproperties=my_font)
    ax[0].set_title("各字體類別預測次數", fontproperties=my_font)
    ax[0].set_xticklabels(new_labels, fontproperties=my_font)
    
    sizes = new_counts
    ax[1].pie(sizes, labels=new_labels, autopct='%1.1f%%', startangle=90, textprops={'fontproperties': my_font}, colors=color_mapping)
    ax[1].axis('equal')  # 让圆饼图保持为圆形
    ax[1].set_title("字體類別分佈", fontproperties=my_font)
    st.pyplot(fig)
