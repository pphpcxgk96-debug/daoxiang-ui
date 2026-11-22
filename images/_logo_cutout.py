# -*- coding: utf-8 -*-
# logo.jpg 白底转透明 logo.png（边界连通背景法 + 内部镂空透明化 + 边缘羽化）
from PIL import Image
import numpy as np
from scipy import ndimage

SRC = r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/logo.jpg"
OUT = r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/logo.png"

img = Image.open(SRC).convert("RGB")
arr = np.asarray(img).astype(np.float32)
H, W = arr.shape[:2]

# 1) 估计背景色（四角取样）
c = 30
corners = np.concatenate([
    arr[:c, :c].reshape(-1, 3), arr[:c, -c:].reshape(-1, 3),
    arr[-c:, :c].reshape(-1, 3), arr[-c:, -c:].reshape(-1, 3)])
bg = corners.mean(axis=0)

# 2) 近背景掩码：与背景色距离近，或纯白高亮低饱和
dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
spread = arr.max(axis=2) - arr.min(axis=2)
near = (dist < 16) | ((lum > 238) & (spread < 14))

# 3) 只保留与边界连通的近背景区 = 主背景（logo 内部镂空不与边界连通）
lbl, n = ndimage.label(near)
border = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
border.discard(0)
subject = ~np.isin(lbl, list(border))

# 4) 内部镂空也透明化：与背景极接近的内部像素（米粒高光 dist>20 不受影响）
inner_white = subject & (dist < 12) & (spread < 10)
subject = subject & ~inner_white

# 5) 闭运算补小孔 + 羽化边缘
subject = ndimage.binary_closing(subject, structure=np.ones((3, 3)))
alpha = ndimage.gaussian_filter(subject.astype(np.float32), 1.0)
alpha = np.clip(alpha, 0, 1)

rgba = np.dstack([arr.astype(np.uint8), (alpha * 255).astype(np.uint8)])
out = Image.fromarray(rgba, "RGBA")
out = out.crop(out.getbbox())
out.save(OUT)
print("logo.png:", out.size)

# 预览：铺在启动页米金底色上检查
prev = Image.new("RGB", (out.size[0] // 2 + 40, out.size[1] // 2 + 40), (219, 200, 143))
small = out.copy()
small.thumbnail((out.size[0] // 2, out.size[1] // 2))
prev.paste(small, (20, 20), small)
prev.save(r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/_logo_preview.png")
print("preview ok")
