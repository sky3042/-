# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
import japanize_matplotlib # 日本語表示のためのライブラリ

def main():
    """
    Streamlitでナダラヤ・ワトソン推定量のバンド幅を調整し、
    結果をインタラクティブにプロットするアプリケーション。
    """
    st.set_page_config(layout="wide")
    st.title('ナダラヤ・ワトソン推定量: バンド幅のインタラクティブ調整')
    
    st.markdown("""
    左側のサイドバーにあるスライダーを動かして、ナダラヤ・ワトソン推定量の**バンド幅 (bandwidth)** を変更できます。
    バンド幅の値によって、回帰曲線がどのように変化するかを確認してみてください。

    - **バンド幅が小さい場合**: 曲線は個々のデータ点に近づき、ギザギザになります（過学習の傾向）。
    - **バンド幅が大きい場合**: 曲線はより滑らかになりますが、データの局所的な特徴を捉えきれなくなる可能性があります（学習不足の傾向）。
    """)

    # --- サイドバー ---
    st.sidebar.header('設定')
    # バンド幅を調整するためのスライダー
    bandwidth = st.sidebar.slider(
        'バンド幅 (bandwidth) を選択',
        min_value=0.1,   # 最小値
        max_value=10.0,  # 最大値
        value=2.0,       # 初期値
        step=0.1         # 刻み幅
    )

    # --- データ定義 ---
    # 販売量 y_i
    y_sales = np.array([
        172, 144, 190, 202, 197, 276, 292, 220, 214, 172, 202, 240, 
        152, 117, 176, 182, 181, 251, 249, 214, 167, 159, 185, 228
    ])
    # 平均気温 x_i
    x_temperature = np.array([
        7.6, 6.0, 9.4, 14.5, 19.8, 22.5, 27.7, 28.3, 25.6, 18.8, 13.3, 8.8, 
        4.9, 6.6, 9.8, 15.7, 19.5, 23.1, 28.5, 26.4, 23.2, 18.7, 13.1, 8.4
    ])

    # --- 計算と描画 ---
    try:
        # ナダラヤ・ワトソン推定量の計算
        kr = KernelReg(endog=y_sales, exog=x_temperature, var_type='c', bw=[bandwidth])

        # 推定結果をプロットするためのx軸の点を生成
        x_grid = np.linspace(x_temperature.min(), x_temperature.max(), 200)

        # 生成したx軸の点に対する予測値を計算
        y_pred, _ = kr.fit(x_grid)

        # Matplotlibでグラフを作成
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. 元のデータの散布図をプロット
        ax.scatter(x_temperature, y_sales, alpha=0.8, label='観測データ', color='royalblue', zorder=2)

        # 2. ナダラヤ・ワトソン推定量による回帰曲線をプロット
        ax.plot(x_grid, y_pred, color='crimson', linewidth=2.5, label=f'ナダラヤ・ワトソン推定量 (バンド幅={bandwidth:.1f})', zorder=3)

        # グラフの装飾
        ax.set_title('平均気温と販売量の関係', fontsize=16)
        ax.set_xlabel('平均気温 ($x_i$)', fontsize=12)
        ax.set_ylabel('販売量 ($y_i$)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax.legend(fontsize=10)
        
        # Streamlitにグラフを表示
        st.pyplot(fig)

    except Exception as e:
        st.error(f"グラフの描画中にエラーが発生しました: {e}")


if __name__ == '__main__':
    main()
