# app_heatmap.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt

st.set_page_config(page_title="洗馬気温推定マップ10m標高補正", layout="wide")

# タイトル（文字サイズを小さくする）
st.markdown(
    "<h3 style='text-align: center; font-size:20px;'>洗馬気温推定マップ 10m標高補正 信大作成</h3>",
    unsafe_allow_html=True
)

st.write("polytemp_10m_YYYYMMDD.csv を読み込んで、気温で色分けしたポイントマップと 0.2℃刻みの等温線を表示します。")

# -------------------------------------------------------
# 観測点リスト（黒丸＋ラベルで表示したい地点）
#   fid, name, Latitude, Longitude, Altitude
# -------------------------------------------------------
poi_data = [
    {"fid": 1,  "name": "KOA1",          "lat": 36.10615778,  "lon": 137.8787694,  "alt": 1035},
    {"fid": 2,  "name": "KOA2",          "lat": 36.10599167,  "lon": 137.8787083,  "alt": 1017},
    {"fid": 3,  "name": "KOA3",          "lat": 36.10616111,  "lon": 137.8790889,  "alt": 1007},
    {"fid": 4,  "name": "KOA4",          "lat": 36.10617778,  "lon": 137.8789667,  "alt": 1005},
     {"fid": 8,  "name": "ondo1_2", "lat": 36.1054,      "lon": 137.8796833,  "alt": 1011},
    {"fid": 9,  "name": "ondo3_4", "lat": 36.10475,     "lon": 137.8803,     "alt": 960},
    {"fid": 10, "name": "ondo5_6", "lat": 36.1041,      "lon": 137.8808167,  "alt": 914},
]
poi_df = pd.DataFrame(poi_data)

# ファイルアップロード
csv_file = st.file_uploader("📂 気温CSV（polytemp_10m_*.csv）を選択", type="csv")

if csv_file is not None:
    # CSV読み込み
    df = pd.read_csv(csv_file)

    st.write("データプレビュー（先頭5行）")
    st.dataframe(df.head())

    # AMD10mDEM アプリから出した CSV を想定
    temp_col = "corrected_Mean air temperature [degC]"

    # 必要な列の存在チェック
    required_cols = ["lat", "lon", temp_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"必要な列が見つかりません: {missing}")
        st.stop()

    # 数値に変換（念のため）
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")

    # 欠損を除去
    df = df.dropna(subset=["lat", "lon", temp_col])

    if df.empty:
        st.warning("有効な座標データがありません（lat/lon/temp が全て欠損になっています）。")
        st.stop()

    # 気温の範囲
    t_min = float(df[temp_col].min())
    t_max = float(df[temp_col].max())
    st.write(f"気温範囲: {t_min:.2f} 〜 {t_max:.2f} ℃")

    # サイドバーで表示パラメータを調整
    st.sidebar.header("表示設定（ポイントマップ）")

    zoom = st.sidebar.slider("ズームレベル", 8, 20, 15, 1)

    # カラースケールの下限・上限（外れ値があればカットできるように）
    vmin, vmax = st.sidebar.slider(
        "カラースケール範囲（℃）",
        min_value=float(np.floor(t_min)),
        max_value=float(np.ceil(t_max)),
        value=(float(np.floor(t_min)), float(np.ceil(t_max))),
        step=0.5,
    )

    radius = st.sidebar.slider("ポイント半径（ピクセル）", 2, 50, 10, 1)

    # 色付け用の正規化
    def temp_to_color(t):
        # vmin〜vmax を 0〜1 に正規化
        if vmax == vmin:
            x = 0.5
        else:
            x = (t - vmin) / (vmax - vmin)
        x = max(0.0, min(1.0, x))  # 0〜1にクリップ

        # シンプルな青→赤グラデーション (R:0→255, G固定, B:255→0)
        r = int(255 * x)
        b = int(255 * (1.0 - x))
        g = 80
        return [r, g, b, 200]  # RGBA

    df["color"] = df[temp_col].apply(temp_to_color)

    # 中心位置
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    st.caption(f"中心座標: lat={center_lat:.6f}, lon={center_lon:.6f}")

    # pydeck のポイントレイヤ
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=radius,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        bearing=0,
    )

    # タブで「ポイントマップ」と「等温線図」を表示
    tab_point, tab_contour = st.tabs(["🟡 ポイントマップ", "📈 等温線（0.2℃刻み）"])

    # -------------------------
    # タブ1: ポイントマップ
    # -------------------------
    with tab_point:
        deck = pdk.Deck(
            layers=[point_layer],
            initial_view_state=view_state,
            map_style=None,  # OpenStreetMap ベース
            tooltip={
                "text": (
                    "lat: {lat}\n"
                    "lon: {lon}\n"
                    f"T: {{{temp_col}}} ℃"
                )
            },
        )

        st.subheader("🟡 気温で色分けしたポイントマップ")
        st.pydeck_chart(deck)

        st.markdown("""
**色の意味（デフォルト設定）**

- 青: カラースケール下限（vmin ℃）付近の低温  
- 赤: カラースケール上限（vmax ℃）付近の高温  
- 黄〜オレンジ: 中間の温度帯
""")

    # -------------------------
    # タブ2: 等温線図（0.2℃刻み）＋観測点
    # -------------------------
    with tab_contour:
        st.subheader("📈 等温線図（0.2℃刻み）＋観測点")

        # 等温線のレベル（0.2℃ごと）
        level_min = np.floor(t_min * 5) / 5.0
        level_max = np.ceil(t_max * 5) / 5.0
        levels = np.arange(level_min, level_max + 0.001, 0.2)

        st.caption(f"等温線レベル: {level_min:.1f} ℃ 〜 {level_max:.1f} ℃ を 0.2℃刻み")

        # tricontourf / tricontour 用の配列
        xs = df["lon"].values
        ys = df["lat"].values
        zs = df[temp_col].values

        fig, ax = plt.subplots(figsize=(6, 5))

        # 塗りつぶし等温線
        cf = ax.tricontourf(xs, ys, zs, levels=levels, cmap="Spectral_r", extend="both")

        # 等温線（線）を上に重ねる
        c_lines = ax.tricontour(xs, ys, zs, levels=levels, colors="k", linewidths=0.5)

        # ラベル（値）を付ける
        ax.clabel(c_lines, inline=True, fontsize=8, fmt="%.1f")

        # --- ここから観測点の黒丸＋ラベルを追加 ---
        poi_lons = poi_df["lon"].values
        poi_lats = poi_df["lat"].values

        # 黒丸プロット（markersize=5 相当; scatter の s は面積なので 5^2=25 を目安）
        ax.scatter(poi_lons, poi_lats, c="k", s=25, marker="o", zorder=10)

        # ラベル表示（name）
        # 少しだけオフセットして文字が点にかぶらないようにする
        dx = (xs.max() - xs.min()) * 0.002
        dy = (ys.max() - ys.min()) * 0.002
        for _, row in poi_df.iterrows():
            ax.text(
                row["lon"] + dx,
                row["lat"] + dy,
                row["name"],
                fontsize=8,
                color="k",
                zorder=11,
            )

        # カラーバー
        cbar = fig.colorbar(cf, ax=ax, label=f"{temp_col} (℃)")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("0.2℃刻み等温線図（観測点付き・簡易平面表示）")
        ax.set_aspect("equal")

        st.pyplot(fig)

        st.markdown("""
※ この等温線図は、緯度・経度をそのまま平面にプロットした簡易表示です。  
観測範囲が数 km〜十数 km 程度なら、形はほぼ問題ないはずです。  
黒丸が観測点（KOA山・おんどとり）、ラベルは name です。
""")

else:
    st.info("polytemp_10m_YYYYMMDD.csv をアップロードしてください。")
