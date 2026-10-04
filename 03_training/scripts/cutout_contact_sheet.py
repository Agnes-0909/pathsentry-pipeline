"""Visual review sheet for train-only object cutouts."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = {0: "person", 1: "vehicle"}


def make_contact_sheet(rows, output: Path):
    chosen = []
    for class_id in CLASS_NAMES:
        class_rows = [row for row in rows if row["class_id"] == class_id]
        if class_rows:
            indexes = np.linspace(0, len(class_rows) - 1, min(24, len(class_rows)), dtype=int)
            chosen.extend(class_rows[i] for i in indexes)
    if not chosen:
        return
    columns, tile_width, tile_height = 6, 176, 176
    sheet = Image.new("RGB", (columns * tile_width, ((len(chosen) + columns - 1) // columns) * tile_height), "#e4e4e4")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        with Image.open(output / row["crop"]) as original:
            crop = original.convert("RGBA")
        crop.thumbnail((tile_width - 12, tile_height - 34))
        x = (index % columns) * tile_width + (tile_width - crop.width) // 2
        y = (index // columns) * tile_height + 3
        sheet.paste(crop, (x, y), crop)
        draw.text(((index % columns) * tile_width + 4, (index // columns) * tile_height + tile_height - 24),
                  f"{CLASS_NAMES[row['class_id']]} {row['source_image']}", fill="black")
    sheet.save(output / "contact_sheet.jpg", quality=90)
