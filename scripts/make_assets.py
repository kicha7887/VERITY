"""Generates assets/logo.png and assets/user.png (run once; outputs are committed)."""
from PIL import Image, ImageDraw

NAVY, GOLD, IVORY = (23, 43, 69), (179, 148, 90), (247, 244, 236)


def logo(path):
    s = 256
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, s - 1, s - 1), 48, fill=NAVY)
    d.ellipse((30, 30, s - 30, s - 30), outline=GOLD, width=7)
    d.polygon([(78, 84), (108, 84), (128, 150), (148, 84), (178, 84), (140, 182), (116, 182)], fill=IVORY)
    d.line((92, 196, 164, 196), fill=GOLD, width=6)
    im.save(path)


def user(path):
    s = 128
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse((0, 0, s - 1, s - 1), fill=GOLD)
    d.ellipse((44, 24, 84, 64), fill=NAVY)
    d.pieslice((24, 66, 104, 146), 180, 360, fill=NAVY)
    im.save(path)


if __name__ == "__main__":
    logo("assets/logo.png"); user("assets/user.png")
