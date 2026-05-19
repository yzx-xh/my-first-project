import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def calculate_metrics(y_true, y_pred):
    """计算 R2、MSE、MAPE。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0
    else:
        mape = float("nan")

    return r2, mse, mape


def _read_text_auto_encoding(file_path):
    """尝试多种编码读取文件内容，返回 (text, encoding)。"""
    for encoding in ("utf-8", "gbk", "gb2312", "utf-8-sig", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "无法识别文件编码，请尝试将文件另存为 UTF-8 格式。"
    )


def load_data(file_path):
    """读取数据文件，自动检测分隔符并跳过中文表头。"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{file_path.resolve()}")

    raw, used_encoding = _read_text_auto_encoding(file_path)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("数据文件为空。")

    first_line = lines[0]
    if "," in first_line:
        sep = ","
    elif "\t" in first_line:
        sep = "\t"
    else:
        sep = r"\s+"

    df = pd.read_csv(file_path, sep=sep, header=None, engine="python", encoding=used_encoding)

    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce")
    if first_row_numeric.isna().any():
        df = df.iloc[1:].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    after_drop = len(df)
    if after_drop == 0:
        raise ValueError("数据文件中无可用的数值数据（去除空值后为空）。")
    if before_drop != after_drop:
        print(f"[提示] 已去除 {before_drop - after_drop} 行包含缺失值的数据，剩余 {after_drop} 行。")

    df.columns = ["Q"] + [f"Y{i}" for i in range(1, df.shape[1])]
    return df


def _build_coeff_names(degree):
    """根据阶数生成系数列名，如 ['a1(Q^3)', 'a2(Q^2)', 'a3(Q)', 'a4']"""
    names = []
    for i in range(degree + 1):
        power = degree - i
        if power > 1:
            names.append(f"a{i + 1}(Q^{power})")
        elif power == 1:
            names.append(f"a{i + 1}(Q)")
        else:
            names.append(f"a{i + 1}")
    return names


def _build_equation(coeffs, degree):
    """根据系数和阶数生成拟合方程字符串。"""
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if power > 1:
            terms.append(f"{c:.6e}*Q^{power}")
        elif power == 1:
            terms.append(f"{c:.6e}*Q")
        else:
            terms.append(f"{c:.6e}")
    return "Y = " + " + ".join(terms)


def fit_polynomial_for_all_y(
    file_path=r"D:\data\原始性能数据-水泵.csv",
    output_path=r"D:\data\拟合结果.csv",
    degree=3,
):
    """
    对第一列 Q 和后续每一列 Y 分别进行多项式拟合。

    Parameters
    ----------
    file_path : str
        输入数据文件路径。
    output_path : str
        输出结果文件路径。
    degree : int
        多项式阶数（默认 3，即三次多项式）。
        2 表示二次，4 表示四次，以此类推。
    """
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree 必须是大于等于 1 的整数。")

    df = load_data(file_path)

    if df.shape[1] < 2:
        raise ValueError("数据至少需要两列（1列 Q + 至少1列 Y）。")

    q_col = df.columns[0]
    y_cols = df.columns[1:]
    Q = df[q_col].values

    if len(Q) < degree + 1:
        raise ValueError(
            f"{degree} 次多项式拟合至少需要 {degree + 1} 个数据点，当前仅有 {len(Q)} 个。"
        )

    coeff_names = _build_coeff_names(degree)
    results = []
    print(f"\n========== {degree} 次多项式拟合结果 ==========\n")

    for y_col in y_cols:
        Y = df[y_col].values
        coeffs = np.polyfit(Q, Y, deg=degree)
        Y_pred = np.polyval(coeffs, Q)
        r2, mse, mape = calculate_metrics(Y, Y_pred)

        row = {"Y列名": y_col}
        for name, coeff in zip(coeff_names, coeffs):
            row[name] = coeff
        row.update({"R2": r2, "MSE": mse, "MAPE(%)": mape})
        results.append(row)

        print(f"--- {y_col} ---")
        print(f"  {_build_equation(coeffs, degree)}")
        print(f"  R2       = {r2:.6f}")
        print(f"  MSE      = {mse:.6e}")
        print(f"  MAPE(%)  = {mape:.6f}")
        print()

    result_df = pd.DataFrame(results)
    print("========== 汇总表格 ==========")
    print(result_df.to_string(index=False))
    print()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存至: {out_path.resolve()}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="对数据文件进行多项式拟合。")
    parser.add_argument(
        "input",
        nargs="?",
        default=r"D:\data\原始性能数据-水泵.csv",
        help=r"输入数据文件路径 (默认: D:\data\原始性能数据.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=r"D:\data\拟合结果.csv",
        help=r"输出结果文件路径 (默认: D:\data\拟合结果.csv)",
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=int,
        default=3,
        help="多项式拟合阶数 (默认: 3，即三次多项式)",
    )
    args = parser.parse_args()

    fit_polynomial_for_all_y(args.input, args.output, degree=args.degree)


if __name__ == "__main__":
    main()
