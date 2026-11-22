# -*- coding: utf-8 -*-
# 从双 IP 合照中抠出左侧小女孩（背景为浅奶油色渐变）
from PIL import Image
import numpy as np
from scipy import ndimage

SRC = r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/_ip_src.jpg"
OUT = r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/ip-girl.png"
HEAD = r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/ip-girl-head.png"
PREVIEW = r"C:/Users/32141/WorkBuddy/2026-08-20-22-50-58/design/images/_ip_preview.png"

img = Image.open(SRC).convert("RGB")
arr = np.asarray(img).astype(np.float32)
H, W = arr.shape[:2]

# 只取左侧 70% 区域（避开右侧米饭小子）
arr_l = arr[:, : int(W * 0.72)]

# 1) 估计背景色（四角取样）
c = 36
corners = np.concatenate([
    arr_l[:c, :c].reshape(-1, 3), arr_l[:c, -c:].reshape(-1, 3),
    arr_l[-c:, :c].reshape(-1, 3), arr_l[-c:, -c:].reshape(-1, 3)])
bg = corners.mean(axis=0)

# 2) 近背景掩码：与背景色距离近，或「高亮度 + 低饱和」像素（应对渐变背景）
dist = np.sqrt(((arr_l - bg) ** 2).sum(axis=2))
lum = 0.299 * arr_l[..., 0] + 0.587 * arr_l[..., 1] + 0.114 * arr_l[..., 2]
spread = arr_l.max(axis=2) - arr_l.min(axis=2)
near = (dist < 18) | ((lum > 234) & (spread < 20))

# 3) 只保留与边界连通的近背景区 = 真背景
lbl, n = ndimage.label(near)
border = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
border.discard(0)
subject = ~np.isin(lbl, list(border))

# 3.5) 矩形预清理：右下米饭团区域（x>0.87宽 且 y>0.67高）强制置为背景
#      （穗粒最下 y≈0.65H、米饭团帽顶 y≈0.685H，该矩形不会伤及麦穗束）
rect_x, rect_y = int(arr_l.shape[1] * 0.87), int(arr_l.shape[0] * 0.67)
subject[rect_y:, rect_x:] = False

# 4) 连通域清理：只保留包含女孩主体（脸部种子点）的连通域，其余碎片全部清除
lbl2, n2 = ndimage.label(subject)
# 种子点：脸部中心（左侧区域中部偏上），必在女孩身上
seed_y, seed_x = int(arr_l.shape[0] * 0.48), int(arr_l.shape[1] * 0.50)
# 若种子点不在前景中，在其附近螺旋搜索最近的 foreground 像素
if not subject[seed_y, seed_x]:
    ys, xs = np.where(subject)
    d2 = (ys - seed_y) ** 2 + (xs - seed_x) ** 2
    idx = int(np.argmin(d2))
    seed_y, seed_x = int(ys[idx]), int(xs[idx])
main_lbl = lbl2[seed_y, seed_x]
mask = (lbl2 == main_lbl)

# 5) 闭运算补小孔 + 羽化边缘
mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
alpha = ndimage.gaussian_filter(mask.astype(np.float32), 1.2)
alpha = np.clip(alpha, 0, 1)

rgba = np.dstack([arr_l.astype(np.uint8), (alpha * 255).astype(np.uint8)])
out = Image.fromarray(rgba, "RGBA")
out = out.crop(out.getbbox())
out.save(OUT)
print("full:", out.size)

# 6) 半身头像版：取上部 62%（草帽 + 脸 + 上身），水平按 alpha 重心裁方；底部 alpha 渐隐防平切
a = np.asarray(out).copy()
ys, xs = np.where(a[..., 3] > 30)
cx = int(xs.mean())
h2 = int(out.size[1] * 0.74)
w2 = h2  # 正方形
x0 = max(0, min(out.size[0] - w2, cx - w2 // 2))
head = out.crop((x0, 0, x0 + w2, h2))
head = head.crop(head.getbbox())
# 底部 8% 高度 alpha 线性渐隐到 0
ha = np.asarray(head).astype(np.float32)
fade = int(head.size[1] * 0.08)
ramp = np.linspace(1.0, 0.0, fade)
ha[-fade:, :, 3] *= ramp[:, None]
head = Image.fromarray(ha.astype(np.uint8), "RGBA")
head.save(HEAD)
print("head:", head.size)

# 7) 预览：铺在纸色底上看效果
prev = Image.new("RGB", (out.size[0] + head.size[0] + 60, max(out.size[1], head.size[1])), (250, 247, 238))
prev.paste(out, (10, 0), out)
prev.paste(head, (out.size[0] + 50, 0), head)
prev.thumbnail((900, 900))
prev.save(PREVIEW)
print("preview ok")
