"""将三种模型已有的训练/测试散点图整合为论文 Fig. 4 风格的 2×3 图。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "results" / "figures"
OUTPUT = FIG_DIR / "05_model_comparison_combined.png"


def get_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def crop_panel(image, side):
    w, h = image.size
    # 去掉每张原图顶部的大标题，保留坐标轴、散点和指标。
    if side == "left":
        box = (int(0.10 * w), int(0.20 * h), int(0.485 * w), int(0.91 * h))
    else:
        box = (int(0.56 * w), int(0.20 * h), int(0.955 * w), int(0.91 * h))
    return image.crop(box)


def main():
    models = [
        ("CatBoost", FIG_DIR / "model_catboost_jointplot.png"),
        ("XGBoost", FIG_DIR / "model_xgboost_jointplot.png"),
        ("LightGBM", FIG_DIR / "model_lightgbm_jointplot.png"),
    ]
    panel_w, panel_h = 700, 500
    header_h, footer_h, gap = 70, 45, 18
    canvas = Image.new("RGB", (3 * panel_w + 4 * gap, header_h + 2 * panel_h + footer_h + 3 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(26, bold=True)
    label_font = get_font(18, bold=True)
    note_font = get_font(14)

    for col, (name, path) in enumerate(models):
        if not path.exists():
            raise FileNotFoundError(path)
        source = Image.open(path).convert("RGB")
        for row, side in enumerate(["left", "right"]):
            panel = crop_panel(source, side)
            panel.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
            x = gap + col * (panel_w + gap) + (panel_w - panel.width) // 2
            y = header_h + gap + row * (panel_h + gap) + (panel_h - panel.height) // 2
            canvas.paste(panel, (x, y))
        x0 = gap + col * (panel_w + gap)
        bbox = draw.textbbox((0, 0), name, font=title_font)
        draw.text((x0 + (panel_w - (bbox[2] - bbox[0])) / 2, 18), name, fill="#222222", font=title_font)

    draw.text((8, header_h + gap + panel_h // 2 - 12), "Training", fill="#222222", font=label_font)
    draw.text((8, header_h + 2 * gap + panel_h + panel_h // 2 - 12), "Testing", fill="#222222", font=label_font)
    draw.text((canvas.width // 2 - 90, canvas.height - footer_h + 8), "Actual value (mg/g)", fill="#555555", font=note_font)

    draw.text((gap, canvas.height - footer_h + 8), "Top row: training set    Bottom row: testing set", fill="#555555", font=note_font)
    draw.text((canvas.width - 350, canvas.height - footer_h + 8), "Observed vs. predicted response", fill="#555555", font=note_font)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(400, 400))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
