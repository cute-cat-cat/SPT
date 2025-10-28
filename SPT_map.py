import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def map(img_path, M=30, visualize=True):
    """
    將 PNG 地圖轉換成 SPT 模型輸入格式 (m, g, x)
    
    參數:
        img_path: 地圖圖片檔案路徑
        M: 輸出地圖大小 (MxM)
        visualize: 是否顯示檢查圖
    
    回傳:
        m: 障礙物二值矩陣 (MxM)
        g: 目標二值矩陣 (MxM)
        x: torch tensor (1, 2, M, M) -> SPT 模型輸入
    """
    # === 1. 讀取圖片 ===
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    # === 2. 建立空地圖 ===
    m = np.zeros(arr.shape[:2])
    g = np.zeros(arr.shape[:2])

    # 障礙物 mask
    obstacle_mask = (arr[:,:,1] > 100) & (arr[:,:,0] < 100)
    goal_mask = (arr[:,:,0] > 200) & (arr[:,:,1] > 200) & (arr[:,:,2] < 100)

    m[obstacle_mask] = 1
    g[goal_mask] = 1

    # === 3. Resize 到 MxM ===
    m_img = Image.fromarray((m*255).astype(np.uint8))
    g_img = Image.fromarray((g*255).astype(np.uint8))

    m_resized = np.array(m_img.resize((M, M), resample=Image.NEAREST))/255.0
    g_resized = np.array(g_img.resize((M, M), resample=Image.NEAREST))/255.0

    # === 4. 轉成 SPT 輸入 tensor ===
    x = np.stack([m_resized, g_resized], axis=0)  # (2, M, M)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 2, M, M)

    # === 5. 視覺化檢查 ===
    if visualize:
        plt.figure(figsize=(10,3))
        plt.subplot(1,3,1)
        plt.imshow(arr)
        plt.title("Original map")

        plt.subplot(1,3,2)
        plt.imshow(m_resized, cmap='gray')
        plt.title("Wall m")

        plt.subplot(1,3,3)
        plt.imshow(g_resized, cmap='hot')
        plt.title("Goal g")
        plt.show()

        print("m shape:", m_resized.shape)
        print("g shape:", g_resized.shape)
        print("障礙物像素數量:", int(m_resized.sum()))
        print("目標點座標:", np.argwhere(g_resized==1))

    return m_resized, g_resized, x

