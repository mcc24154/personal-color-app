import cv2
import numpy as np

# ★★★ Streamlitと連携するための関数 ★★★
def analyze_image_for_color(img_bgr): # ① 引数を使う
    
    if img_bgr is None:
        # Streamlitから画像が渡されなかった場合は何もしない
        return "エラー: 画像データが空です", {}
    
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
    # B > 132 (仮の閾値) => イエローベース (イエベ)
    # B <= 132            => ブルーベース (ブルベ)
    B_THRESHOLD = 132.0 # [cite: 6]
    BASE_COLOR = "イエローベース (イエベ)" if B > B_THRESHOLD else "ブルーベース (ブルベ)" [cite: 6]
    
    # 明度/彩度判定 (L*値で明るさ、a*値で赤み/鮮やかさを補助的に判定)
    # L* > 140 (仮の閾値) => 明るい/クリア (春、夏)
    # L* <= 140           => 暗い/ディープ (秋、冬)
    L_THRESHOLD = 140.0 # [cite: 7]
    BRIGHTNESS = L > L_THRESHOLD 
    
    # 診断の実行
    if BASE_COLOR == "イエローベース (イエベ)":
        if BRIGHTNESS:
            # イエベで明るい (鮮やか/クリア) -> 春
            personal_color = "イエベ春 (Spring)"
        else:
            # イエベで暗い (落ち着いた/ディープ) -> 秋
            personal_color = "イエベ秋 (Autumn)"
    else: # ブルーベース (ブルベ)
        if BRIGHTNESS:
            # ブルベで明るい (涼しげ/ソフト) -> 夏
            personal_color = "ブルベ夏 (Summer)"
        else:
            # ブルベで暗い (はっきり/シャープ) -> 冬
            personal_color = "ブルベ冬 (Winter)" 
    
    # 9. 最終結果の準備
    lab_data = {'L': float(L), 'a': float(A), 'b': float(B)}
    
    # ★★★ 結果を返す (return) ★★★
    return personal_color, lab_data
    
    