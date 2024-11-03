import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
from torchvision import models, transforms
import torch.nn as nn


# 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加載模型

def load_model():
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)  # 加載預訓練權重
    num_classes = 15
    vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes) 

    # 先載入權重
    state_dict = torch.load('best_model_VisionTransformer (1).pth', map_location=torch.device('cpu'))
    vit.load_state_dict(state_dict, strict=False)  # 載入權重，忽略缺失的鍵

    vit.heads.head = torch.nn.Identity()  # 然後用 Identity 替換分類頭

    vit.eval()
    return vit

# 提取特徵
def extract_features(model, image):
    image = transform(image).unsqueeze(0)  # 增加 batch 維度
    with torch.no_grad():
        features = model.forward(image)
    return features.cpu().numpy()

# 加載本地資料，提取特徵並保存
def extract_and_save_local_data(excel_path, save_path):
    features = []
    labels = []
    model = load_model()

    # 讀取兩個工作表
    df1 = pd.read_excel('class (1).xlsx', sheet_name='政府機關')  # 第一個工作表
    df2 = pd.read_excel('class (1).xlsx', sheet_name='公司')  # 第二個工作表

    # 將兩個工作表的數據合併
    df = pd.concat([df1, df2], ignore_index=True)

    for idx, row in df.iterrows():
        image_path = 'AI_good/' + row['類型'] + '/' + row['名稱'] + '/' + row['影像路徑']
        label = row['影像路徑'][0]

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            feature = extract_features(model, image)
            features.append(feature)
            labels.append(label)

    # 保存特徵和標籤
    np.savez(save_path, features=np.array(features), labels=np.array(labels))
    print(f"特徵已保存到 {save_path}")

    print(f"提取的特徵數量: {len(features)}, 提取的標籤數量: {len(labels)}")
    
    if features and labels:  # 確保有數據後再保存
        np.savez(save_path, features=np.array(features), labels=np.array(labels))
        print(f"特徵已保存到 {save_path}")
    else:
        print("沒有提取到任何特徵或標籤，請檢查數據源。")

# 提取本地資料並保存到文件
extract_and_save_local_data('image_files_with_paths.xlsx', 'local_features.npz')