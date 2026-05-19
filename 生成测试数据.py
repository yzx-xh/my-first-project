"""
生成 TWGC 测试数据集
包含：正常数据、时间漂移、越界样本、IQR 异常、分布不均
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2024)

# ===================== 参数设置 =====================
n_normal = 5000          # 正常样本数
n_out_of_bounds = 80     # 越界样本数
n_iqr_outliers = 60      # IQR 异常样本数

# 理论边界
Q_L, Q_U = 0.0, 100.0    # 流量
H_L, H_U = 0.0, 50.0     # 扬程

# 时间范围
t_start = pd.Timestamp("2023-06-01")
t_end = pd.Timestamp("2024-06-01")

# ===================== 1. 生成正常数据（带时间漂移） =====================
timestamps = pd.date_range(t_start, t_end, periods=n_normal)

# 时间漂移：早期偏向低流量/低扬程，后期偏向高流量/高扬程
t_norm = np.linspace(0, 1, n_normal)
Q_shift = 30 * t_norm           # 流量均值随时间右移
H_shift = 10 * np.sin(2 * np.pi * t_norm)  # 扬程周期性漂移

# 分布不均：80% 样本集中在 [20,60]×[10,30]，20% 均匀散布
n_dense = int(n_normal * 0.8)
n_sparse = n_normal - n_dense

Q_dense = np.random.normal(40, 10, n_dense)
H_dense = np.random.normal(20, 5, n_dense)

Q_sparse = np.random.uniform(Q_L, Q_U, n_sparse)
H_sparse = np.random.uniform(H_L, H_U, n_sparse)

Q_normal = np.concatenate([Q_dense, Q_sparse])
H_normal = np.concatenate([H_dense, H_sparse])

# 加入时间漂移
Q_normal = Q_normal + Q_shift + np.random.normal(0, 3, n_normal)
H_normal = H_normal + H_shift + np.random.normal(0, 2, n_normal)

# 裁剪到合理范围（越界样本单独生成）
Q_normal = np.clip(Q_normal, Q_L + 1, Q_U - 1)
H_normal = np.clip(H_normal, H_L + 1, H_U - 1)

# 目标变量：功率 P = 0.02*Q^1.5 + 0.5*H + noise
P_normal = 0.02 * (Q_normal ** 1.5) + 0.5 * H_normal + np.random.normal(0, 5, n_normal)
# 效率 eta = -0.0001*(Q-50)^2 - 0.001*(H-25)^2 + 0.85 + noise
eta_normal = -0.0001 * (Q_normal - 50)**2 - 0.001 * (H_normal - 25)**2 + 0.85
eta_normal += np.random.normal(0, 0.02, n_normal)
eta_normal = np.clip(eta_normal, 0.3, 0.9)

df_normal = pd.DataFrame({
    "timestamp": timestamps,
    "流量_Q": Q_normal,
    "扬程_H": H_normal,
    "功率_P": P_normal,
    "效率_eta": eta_normal,
})

# ===================== 2. 注入越界样本 =====================
t_oob = pd.date_range(t_start, t_end, periods=n_out_of_bounds)
Q_oob = np.concatenate([
    np.random.uniform(-20, Q_L - 0.1, n_out_of_bounds // 2),   # 低于下界
    np.random.uniform(Q_U + 0.1, 150, n_out_of_bounds - n_out_of_bounds // 2)  # 高于上界
])
H_oob = np.random.uniform(H_L, H_U, n_out_of_bounds)
P_oob = 0.02 * np.clip(Q_oob, 0, 150) ** 1.5 + 0.5 * H_oob + np.random.normal(0, 3, n_out_of_bounds)
eta_oob = np.random.uniform(0.4, 0.8, n_out_of_bounds)

df_oob = pd.DataFrame({
    "timestamp": t_oob,
    "流量_Q": Q_oob,
    "扬程_H": H_oob,
    "功率_P": P_oob,
    "效率_eta": eta_oob,
})

# ===================== 3. 注入 IQR 异常（在边界内但目标值异常） =====================
t_iqr = pd.date_range(t_start, t_end, periods=n_iqr_outliers)
Q_iqr = np.random.uniform(30, 70, n_iqr_outliers)
H_iqr = np.random.uniform(15, 35, n_iqr_outliers)

# 功率异常：部分极高、部分极低
n_high = n_iqr_outliers // 2
P_iqr = np.concatenate([
    np.random.uniform(200, 300, n_high),      # 异常高功率
    np.random.uniform(-50, 0, n_iqr_outliers - n_high)  # 异常低功率
])
# 效率异常
eta_iqr = np.concatenate([
    np.random.uniform(0.95, 1.1, n_high),     # 不可能的高效率
    np.random.uniform(0.0, 0.15, n_iqr_outliers - n_high)  # 极低的效率
])

df_iqr = pd.DataFrame({
    "timestamp": t_iqr,
    "流量_Q": Q_iqr,
    "扬程_H": H_iqr,
    "功率_P": P_iqr,
    "效率_eta": eta_iqr,
})

# ===================== 合并 & 打乱 & 保存 =====================
df_all = pd.concat([df_normal, df_oob, df_iqr], ignore_index=True)
# 按时间排序（更符合实际采集顺序）
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

# 保存
output_path = Path(r"D:\python\测试数据.csv")
df_all.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"测试数据已生成: {output_path.resolve()}")
print(f"总样本数: {len(df_all)}")
print(f"  - 正常样本: {n_normal}")
print(f"  - 越界样本: {n_out_of_bounds}")
print(f"  - IQR 异常: {n_iqr_outliers}")
print("\n数据预览:")
print(df_all.head(10).to_string(index=False))
print("\n...")
print(df_all.tail(5).to_string(index=False))
print("\n统计摘要:")
print(df_all.describe().to_string())
