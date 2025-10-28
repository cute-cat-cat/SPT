import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import SPT_map
import Dijkstra演算法

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 建立SPT模組

#Encoder

class SPTEncoder(nn.Module):
    def __init__(self, in_channels=2, d_model=64):  
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels, d_model, kernel_size=1) # 輸入地圖M*M通道為2，輸出高維64
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=1) # 提升特徵抽象度
        self.layer_norm = nn.LayerNorm(d_model)  #正規化，穩定訓練

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        B, C, H, W = x.shape #輸出調整，方便後續展平，H=W=M
        x = x.permute(0, 2, 3, 1).contiguous() #[B,H,W,C]
        x = x.view(B, H * W, C) # 展平
        x = self.layer_norm(x)
        return x

# Transformer

class Transformer(nn.Module):
    def __init__(self, embed_dim=64, num_layers=5, num_heads=8, ff_dim=512, M=30): 
                # (格子的維度,transformer層數,多頭注意力數,FC層數,地圖大小)
        super().__init__()
        self.M = M
        self.pos_encoding = self.positional_encoding(embed_dim, M) 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
        #這項就是傳統transformer的encoder在做的事，包含self-attention, NL,FFN
    #建立PE
    def positional_encoding(self, d, M): 
        pe = torch.zeros(M*M, d)  #建立矩陣
        pos = torch.arange(M*M).unsqueeze(1) #索引位置，unsqueeze(1)=>shape(M*M,1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(M*M)/d))  #頻率縮放項，對應論文公式
        pe[:, 0::2] = torch.sin(pos * div_term)  #sin波
        pe[:, 1::2] = torch.cos(pos * div_term)  #cos波
        return pe  #x shape = (B,M^2, d)
    def forward(self, x):
        x = x + self.pos_encoding.to(x.device)  # x=x+PE
        x = self.transformer(x)            # (B, M^2, d)
        return x
# Decoder
class SPTDecoder(nn.Module):
    def __init__(self, embed_dim=64, M=30):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1) # 類全連階層，拉成一條線(值)
        self.M = M

    def forward(self, x):
        y = self.linear(x)           # (B, M^2, 1)
        y = y.view(-1, self.M, self.M)  # reshape 成距離圖 (B, M, M)
        return y
    
# Encoder + Transformer +Decoder
class SPT(nn.Module):
    def __init__(self, M=30, embed_dim=64):
        super().__init__()
        self.encoder = SPTEncoder(in_channels=2, d_model=embed_dim)
        self.transformer = Transformer(embed_dim=embed_dim, M=M)
        self.decoder = SPTDecoder(embed_dim=embed_dim, M=M)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        y_hat = self.decoder(x)
        return y_hat


M = 30
model = SPT(M=M).to(device)  # <-- GPU：把模型移到 device
criterion = nn.MSELoss()  #使用MSE做Loss訓練
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

# 讀取 map
m, g, x = SPT_map.map("map2.png", M=M) 
y_true_np = Dijkstra演算法.generate_distance_map(m,g)  #正確答案
y_true = torch.tensor(y_true_np, dtype=torch.float32).unsqueeze(0)  #轉成與y_pred同一維度
x = x.to(device)           # <-- GPU
y_true = y_true.to(device) # <-- GPU

# 訓練示範
# ========================== 訓練流程 ==========================
model.train()
num_epochs = 20000
best_loss = float('inf')
snapshots = {}  # 用來存不同階段的結果

for epoch in range(num_epochs):
    optimizer.zero_grad() 
    y_pred = model(x) 
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

    # 更新最佳 loss
    if loss.item() < best_loss:
        best_loss = loss.item()

    # 每 10 epoch 顯示一次 loss
    if (epoch+1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    # 儲存初期 / 中期 / 後期的模型輸出
    if epoch == 0:  # 初期
        snapshots["early"] = y_pred.detach().clone() #記錄第一次未開始訓練的圖形
    elif epoch == num_epochs // 2:  # 中期
        snapshots["mid"] = y_pred.detach().clone()  #紀錄訓練一半時的圖形
    elif epoch == num_epochs - 1:  # 後期
        snapshots["late"] = y_pred.detach().clone()  #紀錄訓練完成的圖形

print(f"\n✅ 訓練結束，最低 Loss: {best_loss:.6f}")

# 作圖
model.eval()
with torch.no_grad():
    y_hat = model(x)

plt.figure(figsize=(14,6))

# 正確答案
plt.subplot(1,4,1)
y_true_plot = y_true.squeeze(0).cpu().numpy()
y_true_plot = (y_true_plot - y_true_plot.min()) / (y_true_plot.max() - y_true_plot.min())
plt.imshow(y_true_plot, cmap='plasma', origin='upper')
plt.title("Ground Truth (Dijkstra)")
plt.colorbar(fraction=0.046, pad=0.04)
goal_coords = np.argwhere(g==1)
for (i,j) in goal_coords:
    plt.scatter(j, i, c='yellow', s=50, marker='o')
plt.axis('off')

# 初期
plt.subplot(1,4,2)
y_early = snapshots["early"].squeeze(0).cpu().numpy()
y_early = (y_early - y_early.min()) / (y_early.max() - y_early.min())
plt.imshow(y_early, cmap='plasma', origin='upper')
plt.title("Early Stage Prediction")
plt.colorbar(fraction=0.046, pad=0.04)
for (i,j) in goal_coords:
    plt.scatter(j, i, c='yellow', s=50, marker='o')
plt.axis('off')

# 中期
plt.subplot(1,4,3)
y_mid = snapshots["mid"].squeeze(0).cpu().numpy()
y_mid = (y_mid - y_mid.min()) / (y_mid.max() - y_mid.min())
plt.imshow(y_mid, cmap='plasma', origin='upper')
plt.title("Mid Stage Prediction")
plt.colorbar(fraction=0.046, pad=0.04)
for (i,j) in goal_coords:
    plt.scatter(j, i, c='yellow', s=50, marker='o')
plt.axis('off')

# 後期
plt.subplot(1,4,4)
y_late = snapshots["late"].squeeze(0).cpu().numpy()
y_late = (y_late - y_late.min()) / (y_late.max() - y_late.min())
plt.imshow(y_late, cmap='plasma', origin='upper')
plt.title("Late Stage Prediction (Final)")
plt.colorbar(fraction=0.046, pad=0.04)
for (i,j) in goal_coords:
    plt.scatter(j, i, c='yellow', s=50, marker='o')
plt.axis('off')
plt.tight_layout()
plt.show()
