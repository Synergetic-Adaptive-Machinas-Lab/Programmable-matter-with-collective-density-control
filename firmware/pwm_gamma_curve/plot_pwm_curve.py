import math
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 750
margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 70
plot_w = W - margin_l - margin_r
plot_h = H - margin_t - margin_b

PWM_MAX = 4095
X_MAX = 7.0

def pwm_curve(mag, gamma):

    t = mag / 7.0
    if abs(gamma) < 1e-6:
        frac = t
    else:
        frac = (1 - math.exp(-gamma * t)) / (1 - math.exp(-gamma))
    return frac * PWM_MAX

def signed_pwm(intensity, gamma):

    sign = 1.0 if intensity >= 0 else -1.0
    return sign * pwm_curve(abs(intensity), gamma)

def to_px(x, y):
    px = margin_l + ((x + X_MAX) / (2 * X_MAX)) * plot_w
    py = margin_t + (1 - (y + PWM_MAX) / (2 * PWM_MAX)) * plot_h
    return (px, py)

def main():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    try:
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        font = ImageFont.truetype(font_path, 16)
        font_small = ImageFont.truetype(font_path, 13)
        font_title = ImageFont.truetype(font_path, 20)
    except Exception:
        font = font_small = font_title = ImageFont.load_default()

    d.text((W / 2, 25), "PWM_EXP_GAMMA vs signed intensity -> PWM mapping curve",
            fill="black", font=font_title, anchor="mm")

    d.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h],
                outline="black", width=2)

    for i in range(-7, 8):
        yv = i * (PWM_MAX / 7)
        x0, y0 = to_px(-X_MAX, yv)
        x1, _ = to_px(X_MAX, yv)
        d.line([x0, y0, x1, y0], fill=(230, 230, 230), width=1)
        d.text((margin_l - 10, y0), f"{int(round(yv))}", fill="black", font=font_small, anchor="rm")

    for i in range(-7, 8):
        x0, y0 = to_px(i, -PWM_MAX)
        _, y1 = to_px(i, PWM_MAX)
        d.line([x0, y1, x0, y0], fill=(240, 240, 240), width=1)
        d.text((x0, margin_t + plot_h + 10), f"{i}", fill="black", font=font_small, anchor="mm")

    zx, _ = to_px(0, 0)
    _, zy = to_px(0, 0)
    d.line([zx, margin_t, zx, margin_t + plot_h], fill=(160, 160, 160), width=1)
    d.line([margin_l, zy, margin_l + plot_w, zy], fill=(160, 160, 160), width=1)

    d.text((margin_l + plot_w / 2, H - 25), "signed intensity  (-7 ~ +7)",
            fill="black", font=font, anchor="mm")

    ylabel = Image.new("RGBA", (260, 30), (255, 255, 255, 0))
    ImageDraw.Draw(ylabel).text((0, 0), "signed PWM value (-4095 ~ +4095)", fill="black", font=font)
    ylabel = ylabel.rotate(90, expand=True)
    img.paste(ylabel, (15, int(margin_t + plot_h / 2 - 100)), ylabel)

    gammas = [-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0]
    colors = {
        -8.0: (183, 28, 28),
        -5.0: (211, 47, 47),
        -3.0: (244, 81, 30),
        -1.0: (255, 152, 0),
         0.0: (120, 120, 120),
         1.0: (100, 181, 246),
         3.0: (33, 150, 243),
         5.0: (63, 81, 181),
         8.0: (13, 71, 161),
    }
    labels = {
        -8.0: "-8.0  (ease-in, strong)",
        -5.0: "-5.0  (ease-in)",
        -3.0: "-3.0  (ease-in)",
        -1.0: "-1.0  (ease-in, mild)",
         0.0: "0.0   (linear, original)",
         1.0: "1.0   (ease-out, mild)",
         3.0: "3.0   (ease-out, current setting)",
         5.0: "5.0   (ease-out)",
         8.0: "8.0   (ease-out, strong)",
    }

    N = 400
    for gamma in gammas:
        color = colors[gamma]
        pts = [to_px(-X_MAX + 2 * X_MAX * i / N, signed_pwm(-X_MAX + 2 * X_MAX * i / N, gamma))
               for i in range(N + 1)]
        width = 4 if gamma == 3.0 else 2
        d.line(pts, fill=color, width=width, joint="curve")
        marker = "square" if gamma < 0 else "circle"
        for intensity in range(-7, 8):
            if intensity == 0:
                continue
            px, py = to_px(intensity, signed_pwm(intensity, gamma))
            r = 3.5
            if marker == "square":
                d.rectangle([px - r, py - r, px + r, py + r], fill=color)
            else:
                d.ellipse([px - r, py - r, px + r, py + r], fill=color)

    lx, ly = margin_l + 15, margin_t + 15
    lw, lh = 300, 22 * len(gammas) + 16
    d.rectangle([lx, ly, lx + lw, ly + lh], fill="white", outline="black", width=1)
    d.text((lx + lw / 2, ly + 10), "gamma (o = ease-out, sq = ease-in)", fill="black", font=font_small, anchor="mm")
    yy = ly + 26
    for gamma in gammas:
        color = colors[gamma]
        d.line([lx + 12, yy, lx + 42, yy], fill=color, width=4)
        d.text((lx + 52, yy), labels[gamma], fill="black", font=font_small, anchor="lm")
        yy += 22

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pwm_gamma_curves.png")
    img.save(out_path)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
