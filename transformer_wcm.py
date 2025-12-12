# -*- coding: utf-8 -*-
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
import pylab
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('tkagg')
# ======== 配置区 ========
DATA_PATH = r"\Inputt_dataframe_2023_with_pred.xlsx"  # 你的本地文件（xlsx/csv）
SAVE_DIR = Path(r"\Transf_WCM")  # <--- 新增
SHEET_NAME = '3月-11月'
TEST_SIZE = 0.2
RANDOM_SEED = 24
BATCH_SIZE = 256
EPOCHS = 2000
LR = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 固定粗糙度常数（数据中仍会构造占位输入，但WCM里将使用可学习rough）
RMSH_CONST = 1.3224
CL_CONST = 13.12

# 缩放（绘图与指标都在缩放后进行；logit训练也基于此缩放）
SM_SCALE = 0.01

# 目标变换安全常数
EPS = 1e-5

# 绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


# ======== 工具 ========
def cosine_deg(deg_array):
    return np.cos(np.deg2rad(deg_array))


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # 过滤NaN
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0
    R, _ = pearsonr(y_true, y_pred)
    MD = float(np.mean(y_pred - y_true))
    RMSE = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ubRMSE = float(np.sqrt(max(RMSE ** 2 - MD ** 2, 0.0)))
    return R, MD, RMSE, ubRMSE


# ======== 目标变换：logit 空间回归 ========
def to_logit_from_raw_y(y_raw: torch.Tensor) -> torch.Tensor:
    y_scaled = torch.clamp(y_raw * SM_SCALE, EPS, 1 - EPS)
    z = torch.log(y_scaled / (1.0 - y_scaled))
    return z


def from_logit_to_raw_y(z: torch.Tensor) -> torch.Tensor:
    y_scaled = torch.sigmoid(z)
    y_raw = y_scaled / SM_SCALE
    return y_raw


# ======== 绘图函数（规范标签、无标题、指标带单位） ========
def _metric_text(R, MD, RMSE, ubRMSE):  # R 无单位，其余用 m³/m³
    return (f"R={R:.3f}\n"
            f"MD={MD:.3f} m³/m³\n"
            f"RMSE={RMSE:.3f} m³/m³\n"
            f"ubRMSE={ubRMSE:.3f} m³/m³")


def plot_scatter(y_true, y_pred, out_path="scatter_wcm.png", title=None, scale=0.01):
    yt = np.asarray(y_true).ravel() * scale
    yp = np.asarray(y_pred).ravel() * scale
    R, MD, RMSE, ubRMSE = compute_metrics(yt, yp)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=18, alpha=0.7)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], linestyle=":", color="gray", linewidth=1.2)
    ax.set_xlabel("Observed SM (m³/m³)", fontsize=14)
    ax.set_ylabel("Estimated SM (m³/m³)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.text(0.05, 0.95, _metric_text(R, MD, RMSE, ubRMSE), transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"), fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(f"已保存散点图：{out_path}")


def plot_scatter_density(y_true, y_pred, out_path="scatter_density_wcm_wholeHRB.png", title=None, scale=0.01):
    x = np.asarray(y_true).ravel() * scale
    y = np.asarray(y_pred).ravel() * scale
    R, MD, RMSE, ubRMSE = compute_metrics(x, y)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], c='0', linewidth=2, linestyle=':', alpha=0.6)
    sc = ax.scatter(x, y, c=z, s=22, cmap='Spectral_r')
    coeff = np.polyfit(x, y, 1)
    p = np.poly1d(coeff)
    pylab.plot(x, p(x), "r")
    plt.colorbar(sc, ax=ax)
    ax.set_xlabel("Observed SM (m³/m³)", fontsize=16)
    ax.set_ylabel("Estimated SM (m³/m³)", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.text(0.05, 0.95, _metric_text(R, MD, RMSE, ubRMSE), transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"), fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=220)
    print(f"已保存密度散点图：{out_path}")


# ======== Transformer Layer for Sequence Modeling ========
class TransformerHeadWithLearnableRough(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_heads=4, num_layers=4):
        super(TransformerHeadWithLearnableRough, self).__init__()

        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"

        self.attn = nn.TransformerEncoderLayer(
            d_model=input_size,  # input_size = d_model
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(self.attn, num_layers=num_layers)
        self.fc = nn.Linear(input_size, 1)

        # 可学习 rough
        self.learnable_rough = LearnableRoughLayer(input_size=input_size)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(1, 0, 2)  # (T, B, F) for transformer
        x = self.transformer(x)
        x = x.mean(dim=0)  # (B, F) over time

        rough = self.learnable_rough(x)  # (B, 2)
        VC = self.fc(x)                  # (B, 1)
        return VC, rough


# ======== 改进版 FNN（SiLU + 轻Dropout，线性输出） ========
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(FNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.fc3(x)  # 线性输出：C/VC头仍是线性值；SM头输出logit
        return x

# ======== Learnable Rough 层 ========
class LearnableRoughLayer(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LearnableRoughLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)  # 输出2个rough参数（类似 RMSH_CONST, CL_CONST）
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        rough = self.fc2(x)  # (B,2)
        return rough


# ======== WCM 层（使用 learnable rough） ========
def WCM_layer(C, VC, sigma, cos_theta, rough):
    gamma = torch.exp(-2.0 * VC / (cos_theta + 1e-6))
    sigma_veg = C * VC * cos_theta * (1.0 - gamma)
    sigma_soil = (sigma - sigma_veg) / (gamma + 1e-6)
    return torch.cat([sigma_soil, rough], dim=1)  # (B,3)


# ======== Logit空间的加权MSE（尾部重加权） ========
class WeightedLogitMSE(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_hat_logit: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
        # y_true_raw -> logit 真值
        z_true = to_logit_from_raw_y(y_true_raw)
        # 基于 [0,1] 的尺度计算样本权重（尾部更大）
        y_scaled = torch.clamp(y_true_raw * SM_SCALE, EPS, 1 - EPS)
        w = 1.0 + self.alpha * torch.abs(y_scaled - torch.mean(y_scaled))
        loss = torch.mean(w * (y_hat_logit - z_true) ** 2)
        return loss


# ======== 数据准备 ========
def load_dataframe(path, sheet=0):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


def build_arrays_from_df(df):
    df = df.copy()  # 不污染原始df
    df["_rowid"] = np.arange(len(df))

    static_cols = ["elevation", "clay", "sand", "tree", "grass", "crops","shrub", "bare", "urban", "snow", "water",]
    dyn_cols = ["NDVI", "NDWI", "NDMI", "LST_C"]
    need_cols = static_cols + dyn_cols + ["VV_Angle_normal", "angle", "SM_10cm"]

    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    for c in need_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["NDVI", "NDWI", "NDMI"]:
        df[c] = df[c].clip(-1, 1)

    # 仅在需要列上做有效性筛选
    df_clean = df.dropna(subset=need_cols).reset_index(drop=True)

    sta = df_clean[static_cols].values.astype(float)          # (N, 11)
    dyn_step = df_clean[dyn_cols].values.astype(float)        # (N, 4)
    dyn = dyn_step[:, None, :]                                # (N, 1, 4) 这里T=1，如有时序可拼多步
    sigma = df_clean["VV_Angle_normal"].values.astype(float)[:, None]  # (N,1)
    cosv = cosine_deg(df_clean["angle"].values.astype(float))[:, None] # (N,1)

    # 虽然有 learnable rough，但保留常数rough作为占位输入，便于接口一致
    rough = np.tile(np.array([RMSH_CONST, CL_CONST], dtype=float), (len(df_clean), 1))  # (N,2)
    y = df_clean["SM_10cm"].values.astype(float)[:, None]                               # (N,1)

    rowid = df_clean["_rowid"].values.astype(int)  # 原始行号映射
    return sta, dyn, sigma, cosv, rough, y, rowid, df_clean


# ======== 主流程 ========
def main():
    # 固定随机种子（基本可重复）
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 读原始表
    df_all = load_dataframe(DATA_PATH, SHEET_NAME)

    # 基于副本做清洗与构造数组，并保留原始行号映射
    sta, dyn, sigma, cosv, rough, y, rowid, df_clean = build_arrays_from_df(df_all)

    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    def sel(a, I):
        return a[I]

    sta_tr, dyn_tr, sigma_tr, cos_tr, rough_tr, y_tr = sel(sta, tr_idx), sel(dyn, tr_idx), sel(sigma, tr_idx), sel(cosv, tr_idx), sel(rough, tr_idx), sel(y, tr_idx)
    sta_va, dyn_va, sigma_va, cos_va, rough_va, y_va = sel(sta, va_idx), sel(dyn, va_idx), sel(sigma, va_idx), sel(cosv, va_idx), sel(rough, va_idx), sel(y, va_idx)

    # to tensor
    sta_tr = to_tensor(sta_tr).to(DEVICE)
    dyn_tr = to_tensor(dyn_tr).to(DEVICE)
    sigma_tr = to_tensor(sigma_tr).to(DEVICE)
    cos_tr = to_tensor(cos_tr).to(DEVICE)
    rough_tr = to_tensor(rough_tr).to(DEVICE)
    y_tr = to_tensor(y_tr).to(DEVICE)

    sta_va = to_tensor(sta_va).to(DEVICE)
    dyn_va = to_tensor(dyn_va).to(DEVICE)
    sigma_va = to_tensor(sigma_va).to(DEVICE)
    cos_va = to_tensor(cos_va).to(DEVICE)
    rough_va = to_tensor(rough_va).to(DEVICE)
    y_va = to_tensor(y_va).to(DEVICE)

    # DataLoader
    train_ds = data.TensorDataset(dyn_tr, sta_tr, sigma_tr, cos_tr, rough_tr, y_tr)
    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Models
    model_C = FNN(input_size=11, hidden_size=128, output_size=1).to(DEVICE)
    model_VC = TransformerHeadWithLearnableRough(input_size=4, hidden_size=64, num_heads=4, num_layers=4).to(DEVICE)
    model_SM = FNN(input_size=3, hidden_size=128, output_size=1).to(DEVICE)  # 输出为 logit

    # 损失 & 优化器
    crit = WeightedLogitMSE(alpha=2.0)
    optC  = torch.optim.AdamW(model_C.parameters(),  lr=LR, weight_decay=1e-4)
    optVC = torch.optim.AdamW(model_VC.parameters(), lr=LR, weight_decay=1e-4)
    optSM = torch.optim.AdamW(model_SM.parameters(), lr=LR, weight_decay=1e-4)

    # 学习率调度（按验证集Loss降低LR）
    schedC  = torch.optim.lr_scheduler.ReduceLROnPlateau(optC,  mode='min', factor=0.5, patience=20, verbose=False)
    schedVC = torch.optim.lr_scheduler.ReduceLROnPlateau(optVC, mode='min', factor=0.5, patience=20, verbose=False)
    schedSM = torch.optim.lr_scheduler.ReduceLROnPlateau(optSM, mode='min', factor=0.5, patience=20, verbose=False)

    best_va = float("inf")
    best_state = None
    patience = 80
    bad_epochs = 0

    # ====== Training loop ======
    for ep in range(1, EPOCHS + 1):
        model_C.train(); model_VC.train(); model_SM.train()
        running_loss = 0.0

        for dyn_b, sta_b, sig_b, cos_b, rough_b, y_b in train_loader:
            VC, learnable_rough = model_VC(dyn_b)                # (B,1), (B,2)
            C  = model_C(sta_b)                                   # (B,1)
            wcm_in = WCM_layer(C, VC, sig_b, cos_b, learnable_rough)  # (B,3)

            z_hat = model_SM(wcm_in)                              # 作为logit输出
            loss  = crit(z_hat, y_b)                              # 在logit空间的加权MSE

            optC.zero_grad(); optVC.zero_grad(); optSM.zero_grad()
            loss.backward()

            # 梯度裁剪，防止不稳定
            params = list(model_C.parameters()) + list(model_VC.parameters()) + list(model_SM.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optC.step(); optVC.step(); optSM.step()
            running_loss += loss.item() * len(y_b)

        # ----- 验证 -----
        model_C.eval(); model_VC.eval(); model_SM.eval()
        with torch.no_grad():
            VC_va, learnable_rough_va = model_VC(dyn_va)
            C_va  = model_C(sta_va)
            wcm_in_va = WCM_layer(C_va, VC_va, sigma_va, cos_va, learnable_rough_va)
            z_hat_va  = model_SM(wcm_in_va)                       # logit
            # 验证损失在logit域计算，和训练一致
            va_loss   = torch.mean((z_hat_va - to_logit_from_raw_y(y_va))**2).item()

        # 调度器根据验证损失调LR
        schedC.step(va_loss); schedVC.step(va_loss); schedSM.step(va_loss)

        # 提前停止
        if va_loss < best_va - 1e-6:
            best_va = va_loss
            best_state = {"C": model_C.state_dict(), "VC": model_VC.state_dict(), "SM": model_SM.state_dict()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if ep % 50 == 0 or ep == 1:
            avg_train = running_loss / len(train_ds)
            print(f"Epoch {ep:4d} | train_loss(z) {avg_train:.6f} | val_loss(z) {va_loss:.6f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {ep}, best val_loss(z) = {best_va:.6f}")
            break

    # ----- 加载最佳权重 -----
    if best_state is not None:
        model_C.load_state_dict(best_state["C"])
        model_VC.load_state_dict(best_state["VC"])
        model_SM.load_state_dict(best_state["SM"])

        # ====== 保存训练好的模型到指定文件夹（新增）======
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "C_state_dict": model_C.state_dict(),
        "VC_state_dict": model_VC.state_dict(),
        "SM_state_dict": model_SM.state_dict(),
        "random_seed": RANDOM_SEED,
        "meta": {
            "desc": "Transformer + WCM（C=FNN, VC=Transformer, SM=FNN）联合训练checkpoint",
            "data_path": DATA_PATH,
            "sheet_name": SHEET_NAME,
            "sm_scale": SM_SCALE,
            "epochs_trained": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        }
    }
    save_path = SAVE_DIR / "transf_wcm_model_wholeHRB.pt"
    torch.save(checkpoint, save_path)
    print(f"已保存训练好的模型到：{save_path}")

    # ====== 评估 & 作图（注意把logit输出反变换到原始单位）======
    model_C.eval(); model_VC.eval(); model_SM.eval()

    # 验证集
    with torch.no_grad():
        VC_va, learnable_rough_va = model_VC(dyn_va)
        C_va  = model_C(sta_va)
        wcm_in_va = WCM_layer(C_va, VC_va, sigma_va, cos_va, learnable_rough_va)
        z_hat_va  = model_SM(wcm_in_va)
        y_hat_va  = from_logit_to_raw_y(z_hat_va)   # 回到原始单位
        y_true_va = y_va.detach().cpu().numpy().ravel()
        y_pred_va = y_hat_va.detach().cpu().numpy().ravel()

    plot_scatter(y_true_va, y_pred_va, out_path="scatter_val_wcm_transformer_logit_3月-11月_wholeHRB.png", scale=SM_SCALE)
    plot_scatter_density(y_true_va, y_pred_va, out_path="scatter_val_density_wcm_transformer_logit_3月-11月_wholeHRB.png", scale=SM_SCALE)

    # 训练集
    with torch.no_grad():
        VC_tr, learnable_rough_tr = model_VC(dyn_tr)
        C_tr  = model_C(sta_tr)
        wcm_in_tr = WCM_layer(C_tr, VC_tr, sigma_tr, cos_tr, learnable_rough_tr)
        z_hat_tr  = model_SM(wcm_in_tr)
        y_hat_tr  = from_logit_to_raw_y(z_hat_tr)
        y_true_tr = y_tr.detach().cpu().numpy().ravel()
        y_pred_tr = y_hat_tr.detach().cpu().numpy().ravel()

    plot_scatter(y_true_tr, y_pred_tr, out_path="scatter_train_wcm_transformer_logit_3月-11月_wholeHRB.png", scale=SM_SCALE)
    plot_scatter_density(y_true_tr, y_pred_tr, out_path="scatter_train_density_wcm_transformer_logit_3月-11月_1_wholeHRB.png", scale=SM_SCALE)

    # ========= 把预测/真值回填到原始文件结构并导出 =========
    # 1) 构造全量长度数组，并按 rowid + 划分索引写回
    # 1) 构造全量长度数组，并按 rowid + 划分索引写回
    N_all = len(df_all)
    pred_all = np.full(N_all, np.nan, dtype=float)
    true_all = np.full(N_all, np.nan, dtype=float)
    split_all = np.array([""] * N_all, dtype=object)

    # 将train/val分别写入（注意：rowid映射到原始df行）
    pred_all[rowid[tr_idx]] = y_pred_tr
    true_all[rowid[tr_idx]] = y_true_tr
    split_all[rowid[tr_idx]] = "train"

    pred_all[rowid[va_idx]] = y_pred_va
    true_all[rowid[va_idx]] = y_true_va
    split_all[rowid[va_idx]] = "val"

    # 2) 仅在当前工作表上追加新列（不改变原有列顺序），其它工作表保持原样
    in_path = Path(DATA_PATH)
    if in_path.suffix.lower() == ".xlsx":
        # 读取整个工作簿的所有工作表
        all_sheets = pd.read_excel(in_path, sheet_name=None)

        # 取出当前SHEET_NAME对应表（若不存在则新建同名空表并对齐行数）
        cur_sheet = all_sheets.get(SHEET_NAME, None)
        if cur_sheet is None:
            # 若原来没有该表，创建一个与 df_all 行数一致的新表（用 df_all 作基）
            cur_sheet = df_all.copy()

        # 为稳妥起见：保证行数一致（若中途做过清洗/筛选，行数通常未变；如不一致，按较短长度截断）
        n = min(len(cur_sheet), N_all)
        # 只对前 n 行回填；若表更长，尾部保持原状
        cur_sheet = cur_sheet.iloc[:n, :].copy()

        # 追加/覆盖三列
        cur_sheet.loc[:n-1, "SM_true"] = true_all[:n]
        cur_sheet.loc[:n-1, "SM_pred"] = pred_all[:n]
        cur_sheet.loc[:n-1, "split"]   = split_all[:n]

        # 写回到一个新的工作簿：原名 + _with_pred.xlsx
        out_path = in_path.with_name(in_path.stem + "_wholeHRB.xlsx")
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            # 先把其它工作表原样写出
            for name, df_sheet in all_sheets.items():
                if name != SHEET_NAME:
                    df_sheet.to_excel(writer, sheet_name=name, index=False)
            # 再写入更新后的当前表
            cur_sheet.to_excel(writer, sheet_name=SHEET_NAME, index=False)

        print(f"✅ 已在保留其它工作表的前提下，更新工作表《{SHEET_NAME}》并导出：{out_path}")

    else:
        # CSV 情况：退化为写出一个 with_pred.csv
        out_path = in_path.with_name(in_path.stem + "_wholeHRB.csv")
        df_csv = df_all.copy()
        df_csv["SM_true"] = true_all
        df_csv["SM_pred"] = pred_all
        df_csv["split"]   = split_all
        df_csv.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 原文件非 xlsx，已导出 CSV：{out_path}")


if __name__ == "__main__":
    main()