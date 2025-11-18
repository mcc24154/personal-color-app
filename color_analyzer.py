import cv2
import numpy as np

# Streamlit側から渡される可能性があるためインポート
try:
    import streamlit as st
    from app_streamlit import to_gal_moji
except (ImportError, ModuleNotFoundError):
    # Streamlit環境外で実行される場合のためのフォールバック
    def to_gal_moji(text):
        return text

# ★★★ Streamlitと連携するための関数 ★★★
def analyze_image_for_color(img_bgr): # ① 引数を使う
    
    if img_bgr is None:
        # Streamlitから画像が渡されなかった場合はエラーを返す (戻り値の形式変更)
        return to_gal_moji("エラー: 画像データが空です"), {}, {}
    
    # 3. 座標の定義を画像サイズ取得後に行う
    h, w, _ = img_bgr.shape
    WHITE_PAPER_SIZE = 100
    SKIN_SIZE = 100
    
    # 【座標定義】撮影ルールに合わせて調整してください（例: 右下隅を狙う）
    WHITE_PAPER_X = w - WHITE_PAPER_SIZE - 50 # 右から50ピクセル内側
    WHITE_PAPER_Y = h - WHITE_PAPER_SIZE - 50 # 下から50ピクセル内側
    
    SKIN_X = w // 2 - 50
    SKIN_Y = h // 2 - 50
    # -------------------------------------
    
    # 1. 白い紙の領域（今回はBGR値で処理）を抽出
    white_patch = img_bgr[
        WHITE_PAPER_Y : WHITE_PAPER_Y + WHITE_PAPER_SIZE,
        WHITE_PAPER_X : WHITE_PAPER_X + WHITE_PAPER_SIZE
    ]
    
    # 2. 白い紙の平均色（BGR）を計算
    mean_bgr_white = np.mean(white_patch, axis=(0, 1))
    
    # 3. 補正係数の計算
    ideal_white_value = 255.0
    b_ratio = ideal_white_value / mean_bgr_white[0]
    g_ratio = ideal_white_value / mean_bgr_white[1]
    r_ratio = ideal_white_value / mean_bgr_white[2]
    
    # 補正係数の上限を設定
    MAX_RATIO = 1.5
    b_ratio = min(b_ratio, MAX_RATIO)
    g_ratio = min(g_ratio, MAX_RATIO)
    r_ratio = min(r_ratio, MAX_RATIO)
    
    # 4. 画像全体に補正を適用（カスタムホワイトバランス）
    img_corrected = np.zeros_like(img_bgr, dtype=np.float32)
    img_corrected[:,:,0] = img_bgr[:,:,0] * b_ratio  # Blueチャンネル補正
    img_corrected[:,:,1] = img_bgr[:,:,1] * g_ratio  # Greenチャンネル補正
    img_corrected[:,:,2] = img_bgr[:,:,2] * r_ratio  # Redチャンネル補正
    
    img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)
    
    # -----------------
    # 5. 補正後の肌色の抽出
    h, w, _ = img_corrected.shape
    
    # 肌の領域も今回は固定の座標で指定（例: 画像の中央）
    SKIN_X = w // 2 - 50
    SKIN_Y = h // 2 - 50
    SKIN_SIZE = 100
    
    skin_patch = img_corrected[SKIN_Y:SKIN_Y+SKIN_SIZE, SKIN_X:SKIN_X+SKIN_SIZE]
    
    # 6. L*a*b*に変換し、平均値を計算（アルゴリズム担当への入力データ）
    img_lab = cv2.cvtColor(skin_patch, cv2.COLOR_BGR2LAB)
    mean_lab = np.mean(img_lab, axis=(0, 1))
    
    # L*, a*, b* 値を変数に格納 (可読性のため)
    L = mean_lab[0]
    A = mean_lab[1]
    B = mean_lab[2]
    
    # =======================================================
    # 8. パーソナルカラー診断ロジック [cite: 6, 7, 8, 9]
    # =======================================================
    
    # L*a*b*のOpenCV範囲: L(0-255), a(0-255), b(0-255, 128付近がニュートラル)
    # 判定閾値は、多くの分析結果に基づいて調整されるべきものであり、ここでは仮の値を設定しています。
    
    # ベースカラー判定 (b*値で黄み/青みを判定)
    B_THRESHOLD = 132.0
    L_THRESHOLD = 140.0

    # --- 連続的なスコア計算 ---
    # B値 (黄み/青み) のスコア
    # 閾値の±8の範囲でグラデーション (例: 124-140)
    B_RANGE_WIDTH = 16 
    b_low_bound = B_THRESHOLD - B_RANGE_WIDTH / 2
    b_high_bound = B_THRESHOLD + B_RANGE_WIDTH / 2

    yellow_score = 0.0
    if B >= b_high_bound:
        yellow_score = 1.0
    elif B <= b_low_bound:
        yellow_score = 0.0
    else:
        yellow_score = (B - b_low_bound) / B_RANGE_WIDTH
    blue_score = 1.0 - yellow_score

    # L値 (明るさ/暗さ) のスコア
    # 閾値の±10の範囲でグラデーション (例: 130-150)
    L_RANGE_WIDTH = 20 
    l_low_bound = L_THRESHOLD - L_RANGE_WIDTH / 2
    l_high_bound = L_THRESHOLD + L_RANGE_WIDTH / 2

    bright_score = 0.0
    if L >= l_high_bound:
        bright_score = 1.0
    elif L <= l_low_bound:
        bright_score = 0.0
    else:
        bright_score = (L - l_low_bound) / L_RANGE_WIDTH
    dark_score = 1.0 - bright_score

    # 各シーズンの相対スコアを計算
    spring_raw_score = yellow_score * bright_score
    autumn_raw_score = yellow_score * dark_score
    summer_raw_score = blue_score * bright_score
    winter_raw_score = blue_score * dark_score

    # 合計スコアで正規化してパーセンテージを算出
    total_raw_score = spring_raw_score + autumn_raw_score + summer_raw_score + winter_raw_score

    season_percentages = {
        to_gal_moji("イエベ春 (Spring)"): 0.0,
        to_gal_moji("イエベ秋 (Autumn)"): 0.0,
        to_gal_moji("ブルベ夏 (Summer)"): 0.0,
        to_gal_moji("ブルベ冬 (Winter)"): 0.0,
    }

    if total_raw_score > 0: # ゼロ除算を避ける
        season_percentages[to_gal_moji("イエベ春 (Spring)")] = round((spring_raw_score / total_raw_score) * 100, 1)
        season_percentages[to_gal_moji("イエベ秋 (Autumn)")] = round((autumn_raw_score / total_raw_score) * 100, 1)
        season_percentages[to_gal_moji("ブルベ夏 (Summer)")] = round((summer_raw_score / total_raw_score) * 100, 1)
        season_percentages[to_gal_moji("ブルベ冬 (Winter)")] = round((winter_raw_score / total_raw_score) * 100, 1)

    # 最も高いパーセンテージのシーズンを主要な診断結果とする
    primary_season = max(season_percentages, key=season_percentages.get)
    
    # 9. 最終結果の準備
    lab_data = {'L': float(L), 'a': float(A), 'b': float(B)}
    
    # ★★★ 結果を返す (return) ★★★
    return primary_season, lab_data, season_percentages
    
    