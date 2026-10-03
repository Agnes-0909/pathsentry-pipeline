"""Render selected LabelMe annotations as compact documentation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "drivable_area": (24, 184, 139),
    "person": (238, 93, 84),
    "vehicle": (255, 197, 75),
}


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render(image_path: Path, label_path: Path, output_path: Path) -> None:
    record = json.loads(label_path.read_text(encoding="utf-8"))
    if record.get("imagePath") != image_path.name:
        raise ValueError(f"Image reference does not match: {label_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    polygons = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill = ImageDraw.Draw(polygons)
    for shape in record.get("shapes", []):
        if shape.get("shape_type") != "polygon" or shape.get("label") != "drivable_area":
            continue
        points = [tuple(point) for point in shape["points"]]
        if len(points) >= 3:
            fill.polygon(points, fill=(*COLORS["drivable_area"], 70))
    image = Image.alpha_composite(image.convert("RGBA"), polygons)
    draw = ImageDraw.Draw(image)

    for shape in record.get("shapes", []):
        if shape.get("shape_type") == "polygon" and shape.get("label") == "drivable_area":
            points = [tuple(point) for point in shape["points"]]
            if len(points) >= 3:
                draw.line(points + points[:1], fill=(*COLORS["drivable_area"], 255), width=5, joint="curve")
        elif shape.get("shape_type") == "rectangle" and shape.get("label") in COLORS:
            color = COLORS[shape["label"]]
            (x1, y1), (x2, y2) = shape["points"]
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            draw.rectangle((left, top, right, bottom), outline=(*color, 255), width=6)
            label = shape["label"]
            label_font = font(26)
            text_box = draw.textbbox((0, 0), label, font=label_font)
            label_width = text_box[2] + 18
            label_height = text_box[3] - text_box[1] + 12
            label_top = max(0, int(top) - label_height)
            draw.rectangle((left, label_top, left + label_width, label_top + label_height), fill=(*color, 255))
            draw.text((left + 9, label_top + 5), label, font=label_font, fill=(18, 29, 34, 255))

    image = image.convert("RGB")
    image.thumbnail((960, 816), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=88, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_dir", type=Path, help="directory containing same-stem JPG and LabelMe JSON files")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("stems", nargs="+", help="image stems to render")
    args = parser.parse_args()
    for stem in args.stems:
        render(args.pairs_dir / f"{stem}.jpg", args.pairs_dir / f"{stem}.json", args.output_dir / f"{stem}.jpg")
        print(args.output_dir / f"{stem}.jpg")


if __name__ == "__main__":
    main()
