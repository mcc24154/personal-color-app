import streamlit as st
import cv2
import numpy as np
import os
import base64
import traceback

from color_analyzer import analyze_image_for_color

# --- 1. 定数定義 ---
FONT_FILE_PATH = "fonts/custom_font.ttf" 
FONT_NAME = "CustomAppFont"

# --- 2. Base64ヘルパー関数の定義 ---
def get_base64_image(image_path):
    print(f"\n=== Base64変換開始 ===")
    print(f"Path: {image_path}")
    
    # ファイルが存在するかチェックし、存在しない場合は空データを返す
    if not os.path.exists(image_path):
        print(f"❌ ファイルが見つかりません: {image_path}")
        return "", "" # Base64データとMIMEタイプを空で返す
        
    try:
        # ファイル読み込み
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Base64変換
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        
        # MIMEタイプ判定
        if ext == ".png":
            mime_type = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"

        print(f"✅ 読み込み成功: MIME={mime_type}, Size={len(img_base64)}文字")
        return img_base64, mime_type

    except Exception as e:
        print(f"❌ Base64変換中に例外が発生しました: {e}")
        traceback.print_exc() 
        return "", ""
    
    # 既存のフォントBase64エンコードロジックを統合
    with open(FONT_FILE_PATH, "rb") as f:
        font_base64 = base64.b64encode(f.read()).decode()
    
    file_ext = os.path.splitext(FONT_FILE_PATH)[1].lower()
    if file_ext == '.otf':
        return font_base64, "opentype"
    else: # .ttf の場合
        return font_base64, "truetype"

# --- 3. フォントCSSのパラメータを取得する関数 ---
def get_font_css_params():
    # FONT_FILE_PATH (例: "fonts/custom_font.ttf") を参照します
    if not os.path.exists(FONT_FILE_PATH):
        return "", ""
    
    try:
        with open(FONT_FILE_PATH, "rb") as f:
            font_base64 = base64.b64encode(f.read()).decode()
            
        # ファイル拡張子からフォント形式を判定
        file_ext = os.path.splitext(FONT_FILE_PATH)[1].lower()
        if 'ttf' in file_ext:
            font_format = "truetype"
        elif 'otf' in file_ext:
            font_format = "opentype"
        else:
            font_format = "truetype" # デフォルト
            
        return font_base64, font_format
    except Exception as e:
        print(f"❌ フォント読み込みエラー: {e}")
        return "", ""
    
# アプリ実行時に一度だけ実行し、グローバル変数に格納
font_base64, font_format = get_font_css_params()

# --- 画像 Base64データのグローバル変数化 ---
# 警告表示は show_start_page() で行うため、ここでは読み込みだけを行う
LOGO_PATH = 'images/app_title_logo.png' 
BG_PATH = 'images/main_visual_start.png' 
#DECO_PATH = 'images/decorative_cosme_01.png'

# 全ての画像データを取得し、グローバル変数として保持
font_base64, font_format = get_font_css_params() # ステップ1で復元
logo_base64, logo_mime = get_base64_image(LOGO_PATH)
bg_base64, bg_mime = get_base64_image(BG_PATH)
#deco_base64, deco_mime = get_base64_image(DECO_PATH)
        

# HTML/CSSアニメーションを定義する関数
def set_cosmetic_flow_css():
    st.markdown(
        """
        <style>
        /* 1. 流れるエリア全体を画面下部に固定 */
        .cosmetic-flow-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 120px; /* 流れるエリアの高さ */
            overflow: hidden;
            pointer-events: none; /* 下のボタンがクリックできるように設定 */
            z-index: 99; /* 他の要素より手前に表示 */
            opacity: 0.8; 
        }
        
        /* 2. 流れる要素の親コンテナ */
        .cosmetic-flow {
            white-space: nowrap; /* 要素を折り返さない */
            animation: flow-right-to-left 40s linear infinite; /* 40秒で無限ループ */
            padding-top: 10px;
        }

        /* 3. 要素をアニメーションさせるキーフレーム */
        @keyframes flow-right-to-left {
            0% {
                transform: translateX(100%); /* 最初は右外側からスタート */
            }
            100% {
                transform: translateX(-100%); /* 左外側まで移動 */
            }
        }
        
        /* 4. 流れる個々の要素（画像など）のスタイル */
        .cosmetic-item {
            display: inline-block;
            width: 100px; /* 画像の幅 */
            height: 100px; /* 画像の高さ */
            margin-right: 50px; /* 要素間のスペース */
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
# CSSを呼び出し、全体に適用する
set_cosmetic_flow_css()

# アプリのメイン実行コードの先頭でこの関数を呼び出す
# set_cosmetic_flow_css()

# --- 1. 定数と設定 ---
st.set_page_config(layout="wide") # 画面を広く使う設定

# --- 背景色を設定するカスタムCSS ---
BACKGROUND_COLOR = "#ffffff"  # ★★★ ここに希望のカラーコード（16進数）を入力 ★★★

st.markdown(
    f"""
    <style>
    /* Streamlitアプリ全体の背景を設定 */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        background-attachment: fixed; /* 背景を固定し、スクロールしても動かないようにする */
    }}
    /* 必要に応じてメインコンテンツ部分の背景も調整 */
    .main .block-container {{
        background-color: transparent; /* メインコンテンツの背景を透明にして、アプリの背景色を透けさせる */
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# -------------------------------------
# 画面の状態管理変数を初期化 
if 'page' not in st.session_state:
    st.session_state.page = 'start' # 初期画面は 'start'
if 'diagnosed_season' not in st.session_state:
    st.session_state.diagnosed_season = None
if 'selected_age' not in st.session_state:
    st.session_state.selected_age = '選択してください'
if 'selected_gender' not in st.session_state:
    st.session_state.selected_gender = '選択してください'


def switch_to_camera():
    # ボタンが押されたときのみ状態を切り替える
    st.session_state['page'] = 'camera'

deco_base64, deco_mime = get_base64_image("images/decorative_cosme_01.png")
import streamlit.components.v1 as components

def show_start_page():
    if not bg_base64 or not logo_base64 or not deco_base64:
        st.error("⚠️ 背景またはロゴ、または装飾画像のBase64データが空です。")
        return

    html_content = f"""
    <div style="
        position: relative;
        width: 100%;
        height: 110vh;
        border-radius: 12px;
        overflow: hidden;
        background-image: url('data:{bg_mime};base64,{bg_base64}');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
    ">
        <img src="data:{logo_mime};base64,{logo_base64}"
            style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 30%;
                max-width: 250px;
                z-index: 10;
            ">

        <img src="data:{deco_mime};base64,{deco_base64}"
            style="position:absolute; top:10%; left:10%; width:60px; animation:float1 3s ease-in-out infinite alternate; z-index:5;">
        <img src="data:{deco_mime};base64,{deco_base64}"
            style="position:absolute; top:15%; right:10%; width:60px; animation:float2 4s ease-in-out infinite alternate; z-index:5;">
        <img src="data:{deco_mime};base64,{deco_base64}"
            style="position:absolute; bottom:10%; left:15%; width:60px; animation:float3 3.5s ease-in-out infinite alternate; z-index:5;">
        <img src="data:{deco_mime};base64,{deco_base64}"
            style="position:absolute; bottom:12%; right:12%; width:60px; animation:float4 4.2s ease-in-out infinite alternate; z-index:5;">
    </div>

    <style>
    @keyframes float1 {{
        0% {{ transform: translateY(0px) rotate(0deg); opacity:1; }}
        100% {{ transform: translateY(-10px) rotate(5deg); opacity:0.95; }}
    }}
    @keyframes float2 {{
        0% {{ transform: translateY(0px) rotate(0deg); opacity:1; }}
        100% {{ transform: translateY(-8px) rotate(-5deg); opacity:0.9; }}
    }}
    @keyframes float3 {{
        0% {{ transform: translateY(0px) rotate(0deg); opacity:1; }}
        100% {{ transform: translateY(-6px) rotate(4deg); opacity:0.92; }}
    }}
    @keyframes float4 {{
        0% {{ transform: translateY(0px) rotate(0deg); opacity:1; }}
        100% {{ transform: translateY(-12px) rotate(-6deg); opacity:0.9; }}
    }}
    </style>
    """

    # ✅ StreamlitのHTMLコンポーネントで出力（これなら確実）
    components.html(html_content, height=300)


    # --- テキストセクション ---
    st.markdown(
        """
        <div style='max-width: 750px; margin: 40px auto 0 auto; text-align: left;'>
            <h2 style='text-align: center;'>肌色分析からあなたにぴったりのカラーパレットを提案！</h2>
            <hr style='margin-top: 20px; margin-bottom: 30px;'>
            <h3 style='color:#444;'>診断ステップ</h3>
            <ol style='line-height: 1.8;'>
                <li>顔写真をアップロード</li>
                <li>自動で肌色を分析</li>
                <li>あなたに似合うカラータイプを判定</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 1. カスタムボタンのCSSを定義 ---
    # ボタンの見た目（背景色、文字色、角丸など）をCSSで定義
    # .stButton > button のセレクタを使ってボタンを装飾
    st.markdown("""
    <style>
    div.stButton > button {
        display: inline-block;
        padding: 14px 40px;
        background-color: #ff8fab; /* カスタムカラー */
        color: white;
        font-size: 18px;
        font-weight: bold;
        text-decoration: none;
        border-radius: 30px;
        transition: 0.2s;
        border: none; /* デフォルトの枠線を消す */
    }
    /* ホバー時の色もCSSで指定 */
    div.stButton > button:hover {
        background-color: #ff6f91;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 2. Streamlitのボタンを配置し、機能を持たせる ---
    # 中央寄せのためのコンテナ
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2: # 真ん中のカラムにボタンを配置して中央寄せにする
        # 前回定義した on_click コールバックを使用
        st.button(
            '診断を始める', 
            on_click=switch_to_camera,
            use_container_width=True 
        )
        
# ----------------------------------------------------------------------

# 各シーズンのおすすめカラーパレット (色見本用)
COLOR_PALETTES = {
    "イエベ春": [
        {"name": "コーラルピンク", "hex": "#F88379"},
        {"name": "ブライトイエロー", "hex": "#FFDB58"},
        {"name": "ターコイズ", "hex": "#40E0D0"},
        {"name": "ライトベージュ", "hex": "#F5F5DC"},
    ],
    "イエベ秋": [
        {"name": "オリーブグリーン", "hex": "#6B8E23"},
        {"name": "テラコッタ", "hex": "#E2725B"},
        {"name": "マスタード", "hex": "#FFD563"},
        {"name": "ダークブラウン", "hex": "#5C4033"},
    ],
    "ブルベ夏": [
        {"name": "スモーキーブルー", "hex": "#8FA9C8"},
        {"name": "ラベンダー", "hex": "#B57EDC"},
        {"name": "パステルピンク", "hex": "#F8BBD0"},
        {"name": "オフホワイト", "hex": "#F0F8FF"},
    ],
    "ブルベ冬": [
        {"name": "ジェットブラック", "hex": "#000000"},
        {"name": "ピュアホワイト", "hex": "#FFFFFF"},
        {"name": "ロイヤルブルー", "hex": "#4169E1"},
        {"name": "フューシャ", "hex": "#FF00FF"},
    ],
}

def generate_color_chips_html(palette):
    """カラーパレットからHTML/CSSのカラーチップを生成する"""
    if not palette:
        return ""

    chips_html = '<div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;">'
    for color in palette:
        text_color = '#FFFFFF' if color['hex'] in ['#000000', '#5C4033', '#6B8E23', '#4169E1'] else '#333333'
        text_shadow = '0 0 2px rgba(0,0,0,0.5)' if color['hex'] == '#FFFFFF' else 'none'
        
        chip = (
            f'<div style="width: 100px; height: 100px; background-color: {color["hex"]}; '
            f'border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); display: flex; '
            f'flex-direction: column; justify-content: flex-end; align-items: center; '
            f'padding: 5px; color: {text_color}; font-size: 12px; font-weight: bold; '
            f'text-shadow: {text_shadow}; border: 1px solid #ccc;">'
            f'{color["name"]}'
            f'</div>'
        )
        chips_html += chip

    chips_html += '</div>'
    return chips_html

def get_text_advice(season_str):
    """診断結果に基づいて文章とカラーパレットによるファッションアドバイスを返す"""
    
    raw_season = season_str.strip()
    full_season = raw_season.split(' ')[0] 
    
    palette = COLOR_PALETTES.get(full_season, [])
    color_chips = generate_color_chips_html(palette)
    
    if full_season == 'イエベ春':
        advice = f"""
        🌸 {full_season} (Spring) のあなたへ
        キーワード:明るさ、軽やかさ、フレッシュ
        ファッションアドバイス:素材はコットンやリネンなど軽やかで自然なものを。
            多色使いも得意なので、柄物や明るいレイヤードで元気な印象を強調しましょう。
            コントラストをつけすぎず、全体を明るくまとめてください。
        """
    elif full_season == 'ブルベ夏':
        advice = f"""
        🌊 {full_season} (Summer) のあなたへ
        キーワード:ソフト、エレガント、涼やか
        ファッションアドバイス:素材はシフォンやレース、シルクなど、軽くて透け感のあるものが得意です。
            優しいトーンでまとめ、グラデーションを使うとよりエレガントに見えます。
            強い色は避け、上品でマットな質感を選ぶのがポイントです。
        """
    elif full_season == 'イエベ秋':
        advice = f"""
        🍁 {full_season} (Autumn) のあなたへ
        キーワード:リッチ、ウォーム、シック
        ファッションアドバイス:素材はツイード、スエード、レザーなど、重厚感のある質感や天然素材を活かしましょう。
            コーディネートはアースカラーを基調に、シックで落ち着いた配色が得意です。
            アクセサリーはゴールドやブロンズなど、マットで光沢の少ないものがおすすめです。
        """
    elif full_season == 'ブルベ冬':
        advice = f"""
        ❄️ {full_season} (Winter) のあなたへ
        キーワード:クリア、シャープ、ドラマティック
        ファッションアドバイス:強いコントラスト（白と黒など）をつけたメリハリのある配色が得意です。
            素材はウールやカシミヤなど、ハリと光沢のあるものがおすすめ。
            シャープなラインや、ミニマルでモダンなデザインが非常によく似合います。
        """
    else:
        return f"""
        ### ❌ 診断結果の特定失敗
        診断結果の文字列 `{season_str.strip()}` から有効な4シーズンを特定できませんでした。
        """

    return f"""
    ### 🎨 おすすめカラーパレット
    {color_chips}
    {advice}
    """


# セッション状態の初期化
if 'diagnosed_season' not in st.session_state:
    st.session_state.diagnosed_season = None
if 'coord_season_key' not in st.session_state:
    st.session_state.coord_season_key = "Winter" # 初期値は冬


def show_diagnosis_page():
    st.header("ステップ1: 写真の撮影")
    st.info("💡 **白い紙**を肌の横に並べ、影が入らないように撮影してください。")
    
    # Webカメラの起動と静止画キャプチャ（Streamlitの強力な機能！）
    captured_file = st.camera_input("カメラで撮影")
    
    if 'diagnosed_season' not in st.session_state:
        st.session_state.diagnosed_season = None
    if 'lab_data' not in st.session_state:
        st.session_state.lab_data = {}
    
    if captured_file is not None:
        st.subheader("ステップ2: カラー分析の実行")
    
    try:
        # 画像処理（成功済み）
        file_bytes = np.asarray(bytearray(captured_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("診断を実行中です..."):
            season, lab_data = analyze_image_for_color(img_bgr) 
            
            st.success(f"🎉 カラー分析ロジックの実行に成功しました！結果: {season}")
            
            # 診断が成功した場合のみ、セッション状態に保存
            st.session_state.diagnosed_season = season
            st.session_state.lab_data = lab_data
            
            # 結果ページへ遷移し、アプリを再実行
            st.session_state.page = 'result'
            st.rerun() # 👈 これで画面が切り替わる
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.error(traceback.format_exc())


def show_simple_camera_page():
    st.header("デバッグ成功：カメラ画面に到達")
    st.markdown("ボタンが正しく機能しました。**ここから元のカメラウィジェットのコードを少しずつ復元します。**")
    

def show_result_page():
    st.title('✅ 診断完了！あなたのパーソナルカラー結果')
    
    # 診断結果がセッションに保存されているか確認 (lines 105-106)
    if st.session_state.diagnosed_season is None:
        st.error("診断結果が見つかりませんでした。もう一度最初からやり直してください。")
        if st.button('やり直す', type='secondary'):
            st.session_state.page = 'start'
            st.rerun()
        return

    # 必須変数の初期化 (line 106)
    diagnosed_text = st.session_state.diagnosed_season 
    full_season_key = diagnosed_text.split(' ')[0].strip()
    season_key = diagnosed_text.split('(')[1].replace(')', '').strip().lower()

    # ----------------------------------------------------
    # 1. 診断結果の即時表示セクション (常に表示される) (line 106)
    # ----------------------------------------------------
    st.success(f"あなたの診断結果は…\n\n## 【 {diagnosed_text} 】です！")
    
    st.subheader("📝 おすすめのファッションアドバイス")
    advice_markdown = get_text_advice(diagnosed_text)
    st.markdown(advice_markdown, unsafe_allow_html=True)
    
    st.subheader("分析された肌色データ (LAB)")
    st.json(st.session_state.lab_data)
    
    
    # ----------------------------------------------------
    # 1.5. ★★★ 選択UIの追加（ここが最も重要）★★★
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("🖼️ コーディネートの条件選択と提案")
    
    col_age, col_gender = st.columns(2)
    
    age_options = ['選択してください', '10代', '20代前半', '20代後半', '30代', '40代', '50代以上']
    gender_options = ['選択してください', '女性', '男性']

    # st.selectboxを配置
    with col_age:
        # keyを設定し、セッション状態に直接書き込む
        st.session_state.selected_age = st.selectbox('あなたの年代', age_options, key="res_age")
    with col_gender:
        st.session_state.selected_gender = st.selectbox('あなたの性別', gender_options, key="res_gender")
        
    st.markdown("---")


    # ----------------------------------------------------
    # 2. コーデ提案の条件付き表示セクション
    # ----------------------------------------------------
    is_info_selected = (st.session_state.selected_age != '選択してください') and \
                    (st.session_state.selected_gender != '選択してください')

    if is_info_selected:
        
        # 性別と年代キーの決定
        if st.session_state.selected_gender == '女性':
            gender_key = 'female'
            
            if st.session_state.selected_age == '50代以上':
                # フォルダ内の画像名に合わせてください（例: 50s-over）
                age_key = '50s' 
            else:
                age_key = st.session_state.selected_age.replace('代前半', 's-early').replace('代後半', 's-late').replace('代', 's')
                
        elif st.session_state.selected_gender == '男性':
            gender_key = 'male'
            age_key = 'all-ages' 
        else:
            gender_key = 'neutral'; age_key = 'general' 

        # 画像パスの生成 (拡張子 .jpg を使用)
        image_filename = f"{season_key}_{age_key}_{gender_key}.jpg"
        image_path = os.path.join("images", image_filename)
        
        st.subheader(f"🎨 {st.session_state.selected_age}{st.session_state.selected_gender}向けコーディネート提案")
        
        # 画像の表示
        if os.path.exists(image_path):
            st.image(image_path, caption=f"【{full_season_key}】に似合うイメージ", width=1000)
        else:
            # ファイル拡張子が .jpg か .png かを最終確認してください。
            st.warning(f"💡 該当の画像は現在準備中です。（検索ファイル名: {image_filename}）")
            
        # Google検索ボタンの表示 (lines 110-112)
        st.markdown("---")
        st.subheader("🔍 その他のイメージを探す")
        search_query = f"{full_season_key} {st.session_state.selected_age} {st.session_state.selected_gender} ファッション"
        base_url = "https://www.google.com/search?tbm=isch&q="
        search_url = base_url + search_query

        st.markdown(
            f'<a href="{search_url}" target="_blank">'
            f'<button style="background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">'
            f'Googleで画像検索する'
            f'</button></a>',
            unsafe_allow_html=True
        )


    else:
        # ★★★ メッセージの修正 ★★★
        st.info("⬆️ コーディネートの提案を見るには、年代と性別を選択してください。")

    # ----------------------------------------------------
    # 3. 画面遷移ボタン (line 113)
    # ----------------------------------------------------
    st.markdown("---")
    if st.button('もう一度診断する', type='secondary'):
        st.session_state.page = 'start'
        st.session_state.diagnosed_season = None 
        st.rerun()
        
        
# ----------------------------------------------------
# ★★★ 最終カスタムCSSの定義と適用（アプリ起動時に実行される）★★★
# ----------------------------------------------------

# 1. フォントCSSの定義
# (font_base64, font_format はファイル先頭でグローバル変数として取得済み)
font_css = f"""
<style>
@font-face {{
    font-family: "{FONT_NAME}";
    src: url("data:font/{font_format};base64,{font_base64}") format("{font_format}");
    font-weight: normal;
    font-style: normal;
}}
html, body, .stApp, .stApp * {{
    font-family: "{FONT_NAME}", sans-serif !important;
}}
</style>
"""

# 2. メインビジュアルCSSの定義 (静的表示用)
# (bg_base64, bg_mime はファイル先頭でグローバル変数として取得済み)
visual_css = f"""
<style>
/* 1. メインビジュアルCSS (背景画像と領域確保) */
.title-visual-container {{
    position: relative;
    width: 100%;
    height: 60vh; /* 高さ確保 */
    overflow: hidden;
    margin: 20px 0;
    /* Base64背景画像がここに適用されます */
    background-image: url("data:{bg_mime};base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    border-radius: 10px;
}}

/* 2. ロゴ画像 (中央に配置) */
.title-logo {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%); 
    width: 60%; 
    max-width: 500px; 
    z-index: 10;
}}

</style>
"""

# 3. 結合と適用 (ここで font_css と visual_css が定義されるため NameError は起きません)
all_custom_css = font_css + visual_css
st.markdown(all_custom_css, unsafe_allow_html=True)

# 画面状態に応じて関数を呼び出す
if st.session_state.page == 'start':
    show_start_page()
elif st.session_state.page == 'camera':
    show_diagnosis_page()
elif st.session_state.page == 'result':
    show_result_page()