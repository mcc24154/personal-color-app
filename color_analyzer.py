import cv2
import numpy as np
from skimage import color

# --- シーズン代表色（改良版） ---
SPRING_COLORS = np.array([
    [75, 8, 20], [80, 10, 25], [70, 5, 15]
])
SUMMER_COLORS = np.array([
    [65, 5, 0], [70, 3, 5], [60, 7, 2]
])
AUTUMN_COLORS = np.array([
    [60, 15, 30], [55, 20, 35], [50, 18, 25]
])
WINTER_COLORS = np.array([
    [55, 0, -10], [60, -5, -5], [65, -2, -15]
])

SEASONS = {
    "Spring": SPRING_COLORS,
    "Summer": SUMMER_COLORS,
    "Autumn": AUTUMN_COLORS,
    "Winter": WINTER_COLORS,
}


def analyze_image_for_color(img_bgr):
    """肌色抽出→LAB平均→4シーズン距離→季節とLAB返却"""

    # ==============================
    # 🟡 ① 肌色領域の抽出（YCrCbマスク）
    # ==============================
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # 肌色の一般的範囲（安定度の高い推奨値）
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)

    skin_pixels = img_bgr[mask > 0]

    if len(skin_pixels) < 50:
        # 肌が全然取れない場合 → 全体で代用（最低限の処理）
        skin_pixels = img_bgr.reshape(-1, 3)

    # ==============================
    # 🔵 ② 肌色を LAB に変換して平均
    # ==============================
    skin_lab = color.rgb2lab(skin_pixels[:, ::-1] / 255.0)  # BGR→RGB
    mean_lab = np.mean(skin_lab, axis=0)

    # ==============================
    # 🔴 ③ 各シーズンとの距離を計算
    # ==============================
    season_distances = {
        season: np.mean(np.linalg.norm(mean_lab - palette, axis=1))
        for season, palette in SEASONS.items()
    }

    # 一番距離が近い季節を選ぶ
    detected_season = min(season_distances, key=season_distances.get)

    # ==============================
    # 🟣 ④ 適合度（％）に正規化
    # ==============================
    inv_scores = {k: 1 / (1 + v) for k, v in season_distances.items()}
    total = sum(inv_scores.values())
    percentages = {k: round((v / total) * 100, 2) for k, v in inv_scores.items()}

    return detected_season, mean_lab, percentages
