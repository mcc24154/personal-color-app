import streamlit as st
import cv2
import numpy as np
import os
import base64
import traceback

from color_analyzer import analyze_image_for_color

# --- ギャル文字変換の定義 ---
GAL_CHAR_MAP = {
    # 50音
    'あ': 'ぁ', 'い': 'ﾚヽ', 'う': 'ぅ', 'え': 'ぇ', 'お': 'ぉ',
    'か': 'ｶゝ', 'き': '､ｷ', 'く': '＜', 'け': 'ﾚﾅ', 'こ': '⊇',
    'さ': '､ﾅ', 'し': '	Ｕ', 'す': 'す', 'せ': 'せ', 'そ': 'ξ',
    'た': 'ﾅﾆ', 'ち': 'ち', 'つ': '⊃', 'て': 'τ', 'と': 'ー⊂',
    'な': 'ﾅょ', 'に': 'ﾚﾆ', 'ぬ': 'ぬ', 'ね': 'ね', 'の': '＠',
    'は': 'ﾚよ', 'ひ': 'ひ', 'ふ': '､ζ､', 'へ': '∧', 'ほ': 'ﾚま',
    'ま': 'ま', 'み': 'ゐ', 'む': ' む', 'め': 'め', 'も': 'も',
    'や': 'ゃ', 'ゆ': 'ゅ', 'よ': 'ょ',
    'ら': ' ら', 'り': '丶)', 'る': 'ゑ', 'れ': 'れ', 'ろ': 'з',
    'わ': 'ゎ', 'ゐ': 'ゐ', 'ゑ': 'ゑ', 'を': 'を', 'ん': 'ω',

    # 濁点・半濁点
    'が': 'ｶゞ', 'ぎ': '､ｷ″', 'ぐ': '＜″', 'げ': 'ﾚﾅ″', 'ご': 'ご',
    'ざ': '､ﾅ″', 'じ': 'Ｕ″', 'ず': 'ず', 'ぜ': 'ぜ', 'ぞ': 'ξ″',
    'だ': 'ﾅﾆ″', 'ぢ': 'ぢ', 'づ': '⊃″', 'で': 'τ″', 'ど': 'ー⊂″',
    'ば': 'ﾚよ″', 'び': 'ひ″', 'ぶ': '､ζ､″', 'べ': '∧″', 'ぼ': 'ﾚま″',
    'ぱ': 'ﾚよ°', 'ぴ': 'ひ°', 'ぷ': '､ζ､°', 'ぺ': '∧°', 'ぽ': 'ﾚま°',

    # 促音・拗音
    'っ': 'っ', 'ゃ': 'ゃ', 'ゅ': 'ゅ', 'ょ': 'ょ',
    'ぁ': 'ぁ', 'ぃ': 'ぃ', 'ぅ': 'ぅ', 'ぇ': 'ぇ', 'ぉ': 'ぉ',
        # カタカナ
    'ア': '了', 'イ': 'イ', 'ウ': '宀', 'エ': '工', 'オ': '才',
    'カ': 'ヵ', 'キ': '≠', 'ク': '勹', 'ケ': 'ヶ', 'コ': '⊃',
    'サ': '廾', 'シ': 'シ', 'ス': 'ス', 'セ': 'セ', 'ソ': '`ﾉ',
    'タ': '勺', 'チ': '于', 'ツ': '〃ﾉ', 'テ': '〒', 'ト': '├',
    'ナ': '＋', 'ニ': '二', 'ヌ': '又', 'ネ': '礻', 'ノ': 'ノ',
    'ハ': '/ヽ', 'ヒ': '匕', 'フ': '┐', 'ヘ': '∧', 'ホ': 'ホ',
    'マ': 'マ', 'ミ': '彡', 'ム': 'ム', 'メ': 'メ', 'モ': 'モ',
    'ヤ': 'ヤ', 'ユ': 'ユ', 'ヨ': '∋',
    'ラ': 'ラ', 'リ': 'リ', 'ル': '儿', 'レ': 'レ', 'ロ': 'ロ',
    'ワ': 'ワ', 'ヰ': 'ヰ', 'ヱ': 'ヱ', 'ヲ': 'ヲ', 'ン': '冫',

    # カタカナ (濁点・半濁点)
    'ガ': 'ヵ″', 'ギ': '≠″', 'グ': '勹″', 'ゲ': 'ヶ″', 'ゴ': '⊃″',
    'ザ': 'ザ', 'ジ': 'ジ', 'ズ': 'ズ', 'ゼ': 'ゼ', 'ゾ': '`ﾉ″',
    'ダ': '勺″', 'ヂ': '于″', 'ヅ': '〃ﾉ″', 'デ': 'デ', 'ド': '├″',
    'バ': '/ヽ″', 'ビ': '匕″', 'ブ': '┐″', 'ベ': '∧″', 'ボ': 'ボ',
    'パ': '/ヽo', 'ピ': '匕o', 'プ': '┐o', 'ペ': '∧o', 'ポ': '木o',

    # カタカナ (促音・拗音)
    'ァ': 'ァ', 'ィ': 'ィ', 'ゥ': 'ゥ', 'ェ': 'ェ', 'ォ': 'ォ',
    'ッ': 'ッ', 'ャ': 'ャ', 'ュ': 'ュ', 'ョ': '∋',
    'ヴ': 'ヴ',
}
    
def to_gal_moji(text):
    if st.session_state.get('language_mode', 'ノーマル') == 'ノーマル':
        return text
    
    return "".join([GAL_CHAR_MAP.get(char, char) for char in text])

def t(text):
    """現在の言語モードに合わせて自動変換（ノーマル/ギャル）"""
    if st.session_state.get("language_mode") == "gal":
        return to_gal_moji(text)
    return text

# --- 1. 定数定義 ---
FONT_FILE_PATH = "fonts/custom_font.ttf" 
FONT_NAME = "CustomAppFont"

# --- 2. Base64ヘルパー関数の定義 ---
import os, base64, traceback

def get_base64_image(image_path):
    """
    Streamlit Cloud 対応版：アプリの実行ディレクトリからの絶対パスで読み込む
    """
    print(f"\n=== Base64変換開始 ===")
    print(f"指定されたパス: {image_path}")

    # ★ Streamlit Cloud でも安全に存在を確認できる絶対パス
    abs_path = os.path.join(os.path.dirname(__file__), image_path)

    print(f"実際に読み込むパス: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"❌ ファイルが見つかりません（Cloud 上）: {abs_path}")
        return "", ""

    try:
        with open(abs_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Base64変換
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        ext = os.path.splitext(abs_path)[1].lower()

        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        print(f"✅ 読み込み成功: MIME={mime_type}, Size={len(img_base64)}文字")
        return img_base64, mime_type

    except Exception as e:
        print(f"❌ Base64変換中に例外発生: {e}")
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
        print(t(f"❌ フォント読み込みエラー: {e}"))
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
    st.session_state.selected_age = t('選択してください')
if 'selected_gender' not in st.session_state:
    st.session_state.selected_gender = t('選択してください')
if 'language_mode' not in st.session_state:
    st.session_state.language_mode = t('ノーマル')
    st.session_state.season_percentages = {}

# --- 言語切り替え ---
mode_label = st.radio(
    "表示モード",
    ["ノーマル", "､ｷ″ゃゑ文字"],
    horizontal=True,
)
# 内部値に統一
if mode_label == "ノーマル":
    st.session_state.language_mode = "normal"
else:
    st.session_state.language_mode = "gal"


def switch_to_camera():
    # ボタンが押されたときのみ状態を切り替える
    st.session_state['page'] = 'camera'

deco1_base64, deco1_mime = get_base64_image("images/decorative_cosme_01.png")
deco8_base64, deco8_mime = get_base64_image("images/decorative_cosme_21.png")
deco9_base64, deco9_mime = get_base64_image("images/decorative_cosme_22.png")
deco10_base64, deco10_mime = get_base64_image("images/decorative_cosme_23.png")
deco11_base64, deco11_mime = get_base64_image("images/decorative_cosme_24.png")

cosme1_base64, cosme1_mime = get_base64_image("images/cosme_flow_01.png")
cosme2_base64, cosme2_mime = get_base64_image("images/cosme_flow_02.png")
cosme3_base64, cosme3_mime = get_base64_image("images/cosme_flow_03.png")
cosme4_base64, cosme4_mime = get_base64_image("images/cosme_flow_04.png")
cosme5_base64, cosme5_mime = get_base64_image("images/cosme_flow_05.png")
cosme6_base64, cosme6_mime = get_base64_image("images/cosme_flow_06.png")
cosme7_base64, cosme7_mime = get_base64_image("images/cosme_flow_07.png")
cosme8_base64, cosme8_mime = get_base64_image("images/cosme_flow_08.png")
cosme9_base64, cosme9_mime = get_base64_image("images/cosme_flow_09.png")
cosme10_base64, cosme10_mime = get_base64_image("images/cosme_flow_10.png")

import streamlit.components.v1 as components

def show_start_page():
    if not bg_base64 or not logo_base64 or \
        not deco1_base64 :
        st.error("⚠️ 画像ファイルの一部が見つからないか、Base64データが空です。ファイルパスを確認してください。")
        return

    html_content = f"""
    <div style="
        position: relative;
        width: 100%;
        height: 500px;
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
                width: 40%;
                max-width: 300px;
                z-index: 10;
            ">

        <img src="data:{deco1_mime};base64,{deco1_base64}"
            style="position:absolute; bottom:0%; left:27%; width:200px; animation:float1 3s ease-in-out infinite alternate; z-index:5;">
        <img src="data:{deco1_mime};base64,{deco1_base64}"
            style="position:absolute; bottom:0%; right:27%; width:200px; animation:float1 3s ease-in-out infinite alternate; z-index:5;">
        <img src="data:{deco8_mime};base64,{deco8_base64}" 
            style="position:absolute; top:3%; right:25%; width:150px;
            animation:float1 3s ease-in-out infinite alternate; z-index:5;">
        <img src="data:{deco9_mime};base64,{deco9_base64}" 
            style="position:absolute; bottom:12%; left:37%; width:120px; 
            animation:blink 1.5s step-end infinite; z-index:5;">
        <img src="data:{deco10_mime};base64,{deco10_base64}" 
            style="position:absolute; top:7%; left:35%; width:100px; 
            animation:blink 1.5s step-end infinite; z-index:5;">
        <img src="data:{deco11_mime};base64,{deco11_base64}" 
            style="position:absolute; bottom:22%; right:32%; width:100px; 
            animation:blink 1.5s step-end infinite; z-index:5;">
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
    @keyframes blink {{
    0% {{ opacity: 1; }} /* 最初は完全に表示 */
    50% {{ opacity: 0; }} /* 半分で完全に透明 */
    100% {{ opacity: 1; }} /* 最後は再び完全に表示 */
    }}
    </style>
    """

    # 画像を全て表示させるため、十分な高さを設定します
    import streamlit.components.v1 as components 
    components.html(html_content, height=600)

    # --- テキストセクション ---
    html_text = f"""
    <div style='max-width: 750px; margin: 40px auto 0 auto; text-align: left;'>
        <h2 style='text-align: center;'>{t("肌色分析からあなたにぴったりのカラーパレットを提案！")}</h2>
        <hr style='margin-top: 20px; margin-bottom: 30px;'>
        <h3 style='color:#444;'>{t("診断ステップ")}</h3>
        <ol style='line-height: 1.8;'>
            <li>{t("顔写真をアップロード")}</li>
            <li>{t("自動で肌色を分析")}</li>
            <li>{t("あなたに似合うカラータイプを判定")}</li>
        </ol>
    </div>
    """
    st.markdown(html_text, unsafe_allow_html=True)
    
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
            t('診断を始める'), 
            on_click=switch_to_camera,
            use_container_width=True 
        )
        
    st.markdown(
        """
        <style>
        /* コスメが流れるためのコンテナ */
        .marquee-container {
            width: 100%;
            white-space: nowrap; /* 画像が折り返さないようにする */
            overflow: hidden;  /* 横の余分なコンテンツを隠す */
            white-space: nowrap; 
            margin: 30px 0;
        }

        /* アニメーションを適用する要素 (コスメ画像全体を格納) */
        .marquee-content {
            display: flex;
            transform: translateY(10px);
            animation: marquee-scroll 70s linear infinite; /* 20秒で無限に流れる */
        }

        /* 流れる動きの定義 */
        @keyframes marquee-scroll {
            0% { transform: translateY(0%); } /* 開始地点 */
            100% { transform: translateX(-100%); } /* コンテンツ幅分左へ移動 */
        }
        </style>
        """,
        unsafe_allow_html=True # st.markdown の場合は必要です
    )
        
    # --- コスメが流れるセクション ---
    cosme_html_content = """
    <div class="marquee-container">
        <div class="marquee-content">
            """
    cosme_images = ""

    # 10枚の画像を1セットとして定義 (これを3回繰り返す)
    image_set = f"""
        <img src="data:{cosme1_mime};base64,{cosme1_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme2_mime};base64,{cosme2_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme3_mime};base64,{cosme3_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme4_mime};base64,{cosme4_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme5_mime};base64,{cosme5_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme6_mime};base64,{cosme6_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme7_mime};base64,{cosme7_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme8_mime};base64,{cosme8_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme9_mime};base64,{cosme9_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
        <img src="data:{cosme10_mime};base64,{cosme10_base64}" style="width: 80px; height: 80px; margin-right: 40px;">
    """

    # 10枚の画像を1セットとして定義 (アニメーション遅延を計算)
    image_set_parts = []

    # cosme1 から cosme10 までの 10枚をループで処理
    for i in range(1, 11): 
        # 垂直アニメーションの遅延を計算: i番目の画像は (i * 0.2秒) 遅れて動き始める
        delay_time = i * 0.2 
        
        # <img> タグに wave-up-down アニメーションと animation-delay を追加
        image_set_parts.append(f"""
            <img src="data:{globals()[f'cosme{i}_mime']};base64,{globals()[f'cosme{i}_base64']}" 
            style="width: 80px; height: 100px; margin-right: 50px;  {delay_time}s;">
        """)

    # 10枚分の HTML 文字列を結合
    image_set = "".join(image_set_parts)

    # 3セット繰り返して連結し、流れる幅を確保
    cosme_images = image_set + image_set + image_set


    # --- その後の cosme_html_content の組み立ては変更なし ---
    cosme_html_content = f"""
    <div class="marquee-container">
        <div class="marquee-content">
            {cosme_images}
        </div>
    </div>
    """

    # st.markdown を st.html に変更する
    st.html(
        cosme_html_content,
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
    
    season_key = season_str.strip().lower()     
    
    palette = COLOR_PALETTES.get(season_key.capitalize(), [])
    color_chips = generate_color_chips_html(palette)
    
    if season_key == 'spring':
        advice = t(
            f"🌸 {season_str} (Spring) のあなたへ\n"
            "キーワード:明るさ、軽やかさ、フレッシュ\n"
            "ファッションアドバイス:素材はコットンやリネンなど軽やかで自然なものを。\n"
            "多色使いも得意なので、柄物や明るいレイヤードで元気な印象を強調しましょう。\n"
            "コントラストをつけすぎず、全体を明るくまとめてください。"
        )
    elif season_key == 'summer':
        advice = t(
            f"🌊 {season_str} (Summer) のあなたへ\n"
            "キーワード:ソフト、エレガント、涼やか\n"
            "ファッションアドバイス:素材はシフォンやレース、シルクなど、軽くて透け感のあるものが得意です。\n"
            "優しいトーンでまとめ、グラデーションを使うとよりエレガントに見えます。\n"
            "強い色は避け、上品でマットな質感を選ぶのがポイントです。"
        )
    elif season_key == 'autumn':
        advice = t(
            f"🍁 {season_str} (Autumn) のあなたへ\n"
            "キーワード:リッチ、ウォーム、シック\n"
            "ファッションアドバイス:素材はツイード、スエード、レザーなど、重厚感のある質感や天然素材を活かしましょう。\n"
            "コーディネートはアースカラーを基調に、シックで落ち着いた配色が得意です。\n"
            "アクセサリーはゴールドやブロンズなど、マットで光沢の少ないものがおすすめです。"
        )
    elif season_key == 'winter':
        advice = t(
            f"❄️ {season_str} (Winter) のあなたへ\n"
            "キーワード:クリア、シャープ、ドラマティック\n"
            "ファッションアドバイス:強いコントラスト（白と黒など）をつけたメリハリのある配色が得意です。\n"
            "素材はウールやカシミヤなど、ハリと光沢のあるものがおすすめ。\n"
            "シャープなラインや、ミニマルでモダンなデザインが非常によく似合います。"
        )
    else:
        return f"""
        ### t(❌ 診断結果の特定失敗
        診断結果の文字列 `{season_str.strip()}` から有効な4シーズンを特定できませんでした。)
        """

    return f"""
    ### t(🎨 おすすめカラーパレット)
    {color_chips}
    {advice}
    """


# セッション状態の初期化
if 'diagnosed_season' not in st.session_state:
    st.session_state.diagnosed_season = None
if 'coord_season_key' not in st.session_state:
    st.session_state.coord_season_key = "Winter" # 初期値は冬


def show_diagnosis_page():

    st.subheader(t("ステップ1: 写真を選ぶ"))

    # --- 画像アップロード or カメラ撮影 ---
    uploaded_image = st.file_uploader(
        t("📁 画像をアップロードしてください (PNG/JPG)"),
        type=["png", "jpg", "jpeg"]
    )

    st.write(t("または ↓"))

    captured_image = st.camera_input(t("📸 カメラで撮影する"))

    # 画像が未入力の場合は処理しない
    if uploaded_image is None and captured_image is None:
        st.info(t("写真をアップロードするか、カメラで撮影してください。"))
        return

    # --- 入力された画像を OpenCV 形式へ変換 ---
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif captured_image is not None:
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- ステップ2: カラー分析 ---
    st.subheader(t("ステップ2: カラー分析の実行"))

    try:
        with st.spinner(t("診断を実行中です...")):
            season, lab_data, season_percentages = analyze_image_for_color(img_bgr)

        st.success(t(f"🎉 カラー分析が完了しました！結果: {season}"))

        # セッションへ保存
        st.session_state.diagnosed_season = season
        st.session_state.lab_data = lab_data
        st.session_state.season_percentages = season_percentages

        st.session_state.page = "result"
        st.rerun()

    except Exception as e:
        st.error(t(f"カラー分析ロジックの実行中にエラーが発生しました。エラー: {e}"))
        st.info(t("画像を撮り直して再度お試しください。"))


def show_result_page():
    st.title(t('✅ 診断完了！あなたのパーソナルカラー結果'))
    
    # 診断結果がセッションに保存されているか確認 (lines 105-106)
    if st.session_state.diagnosed_season is None:
        st.error(t("診断結果が見つかりませんでした。もう一度最初からやり直してください。"))
        if st.button(t('やり直す', type='secondary')):
            st.session_state.page = 'start'
            st.rerun()
        return
    
    # --- レーダーチャート表示（season_percentages を使用） ---
    if "season_percentages" in st.session_state:
        st.subheader(t("シーズン適合度（%）"))
        if st.session_state.season_percentages:
            # パーセンテージを降順にソートして表示
            sorted_percentages = sorted(st.session_state.season_percentages.items(), key=lambda item: item[1], reverse=True)
            for season, percentage in sorted_percentages:
                st.write(f"- {to_gal_moji(season)}: **{percentage:.1f}%**")
                st.progress(int(percentage)) # プログレスバーで視覚的に表示
        else:
            st.info(to_gal_moji(t("各シーズンの適合度データがありません。")))
        
    # 必須変数の初期化 (line 106)
    diagnosed_text = st.session_state.diagnosed_season 
    full_season_key = diagnosed_text.split(' ')[0].strip()
    if '(' in diagnosed_text and ')' in diagnosed_text:
        season_key = diagnosed_text.split('(')[1].replace(')', '').strip().lower()
    else:
        season_key = diagnosed_text.strip().lower()
        
    # ----------------------------------------------------
    # 1. 診断結果の即時表示セクション (常に表示される) (line 106)
    # ----------------------------------------------------
    st.success(t(f"あなたの診断結果は…\n\n## 【 {diagnosed_text} 】です！"))
    
    st.subheader(t("📝 おすすめのファッションアドバイス"))
    advice_markdown = get_text_advice(diagnosed_text)
    st.markdown(advice_markdown, unsafe_allow_html=True)
    
    st.subheader(t("分析された肌色データ (LAB)"))
    lab_LAB = {
        "L": float(st.session_state.lab_data[0]),
        "A": float(st.session_state.lab_data[1]),
        "B": float(st.session_state.lab_data[2]),
    }

    st.json(lab_LAB)
    
    
    # ----------------------------------------------------
    # 1.5. ★★★ 選択UIの追加（ここが最も重要）★★★
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader(t("🖼️ コーディネートの条件選択と提案"))
    
    col_age, col_gender = st.columns(2)
    
    age_options = [t(x) for x in ['選択してください', '10代', '20代前半', '20代後半', '30代', '40代', '50代以上']]
    gender_options = [t(x) for x in ['選択してください', '女性', '男性']]

    # st.selectboxを配置
    with col_age:
        # keyを設定し、セッション状態に直接書き込む
        st.session_state.selected_age = st.selectbox(t('あなたの年代'), age_options, key="res_age")
    with col_gender:
        st.session_state.selected_gender = st.selectbox(t('あなたの性別'), gender_options, key="res_gender")
        
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
        
        st.subheader(t(f"🎨 {st.session_state.selected_age}{st.session_state.selected_gender}向けコーディネート提案"))
        
        # 画像の表示
        if os.path.exists(image_path):
            st.image(image_path, caption=t(f"【{full_season_key}】に似合うイメージ"), width=1000)
        else:
            # ファイル拡張子が .jpg か .png かを最終確認してください。
            st.warning(t(f"💡 該当の画像は現在準備中です。（検索ファイル名: {image_filename}）"))
            
        # Google検索ボタンの表示 (lines 110-112)
        st.markdown("---")
        st.subheader(t("🔍 その他のイメージを探す"))
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
        st.info(t("⬆️ コーディネートの提案を見るには、年代と性別を選択してください。"))

    # ----------------------------------------------------
    # 3. 画面遷移ボタン (line 113)
    # ----------------------------------------------------
    st.markdown("---")
    if st.button(t('もう一度診断する'), type='secondary'):
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
    width: 50% !important;
    height: auto; /* 高さ確保 */
    padding-bottom: 100%;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
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