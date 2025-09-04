# callbacks_uw.py
import math, random, cv2, numpy as np
import torch

# to_numpy, to_tensor, gray_world_wb, lab_clahe, add_haze, blue_green_cast, gamma_contrast, unsharp, copy_paste_masks, on_preprocess_batch

# ---- 유틸 ----
def to_numpy(img_t):  # (B,C,H,W) float32 [0,1] -> list of HWC BGR uint8
    imgs = (img_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    imgs = [cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_RGB2BGR) for img in imgs]
    return imgs

def to_tensor(imgs):  # list of HWC BGR uint8 -> (B,C,H,W) float32 [0,1] RGB
    arr = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2,0,1) for im in imgs]
    t = torch.from_numpy(np.stack(arr)).float() / 255.0
    return t

# ---- 전처리/증강 원자 연산 ----
def gray_world_wb(bgr):
    b,g,r = cv2.split(bgr)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    k = (mb+mg+mr)/3.0 + 1e-6
    b = np.clip(b * (k/mb), 0, 255); g = np.clip(g * (k/mg), 0, 255); r = np.clip(r * (k/mr), 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

def lab_clahe(bgr, clip=2.0):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def add_haze(bgr, alpha=0.2, A=(210, 230, 255)):
    # x*(1-a)+A*a (A는 BGR, 수중 청녹 성분 강조)
    fog = np.full_like(bgr, A, dtype=np.uint8)
    out = cv2.addWeighted(bgr, 1.0-alpha, fog, alpha, 0)
    return out

def blue_green_cast(bgr, s=0.1):
    scale = np.array([1.0+random.uniform(-s,0.05), 1.0+random.uniform(0, s), 1.0+random.uniform(0, s)], dtype=np.float32)
    out = np.clip(bgr.astype(np.float32)*scale, 0, 255).astype(np.uint8)
    return out

def gamma_contrast(bgr, g=1.0, c=1.0):
    x = (bgr.astype(np.float32)/255.0) ** g
    m = x.mean(axis=(0,1), keepdims=True)
    x = np.clip((x - m) * c + m, 0, 1)
    return (x*255).astype(np.uint8)

def unsharp(bgr, k=0.4):
    blur = cv2.GaussianBlur(bgr, (0,0), 3)
    out = cv2.addWeighted(bgr, 1+k, blur, -k, 0)
    return out

def copy_paste_masks(imgs, masks, p=0.2, max_paste=2):
    # imgs: list of HWC uint8(BGR), masks: list of list[np.uint8(HW)](0/1)
    if random.random() > p: return imgs, masks
    n = len(imgs)
    for i in range(n):
        src = random.randrange(n)
        if len(masks[src]) == 0: continue
        cp_cnt = random.randint(1, min(max_paste, len(masks[src])))
        for _ in range(cp_cnt):
            m = masks[src][random.randrange(len(masks[src]))]  # HW(0/1)
            ys, xs = np.where(m>0)
            if len(xs)==0: continue
            h, w = imgs[i].shape[:2]
            tx = random.randint(0, max(0, w-1)); ty = random.randint(0, max(0, h-1))
            # bounding box crop
            x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
            patch = imgs[src][y0:y1+1, x0:x1+1].copy()
            mpatch = m[y0:y1+1, x0:x1+1].astype(bool)
            ph, pw = patch.shape[:2]
            if ph<2 or pw<2: continue
            tx = min(tx, w-pw); ty = min(ty, h-ph)
            roi = imgs[i][ty:ty+ph, tx:tx+pw]
            roi[mpatch] = patch[mpatch]
            imgs[i][ty:ty+ph, tx:tx+pw] = roi
    return imgs, masks

def perlin_like_mask(h, w, scale=0.1):
    # 저주파 random mask (간단 대체): 가우시안 노이즈 → 흐리기
    m = np.random.randn(h, w).astype(np.float32)
    k = int(max(3, min(h,w) * scale)) | 1
    m = cv2.GaussianBlur(m, (k,k), k*0.5)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return m  # [0,1]

def beer_lambert_underwater(bgr, A=(210,232,255)):
    H,W = bgr.shape[:2]
    d = perlin_like_mask(H,W, scale=0.06)  # pseudo depth
    # 채널별 감쇠(수중은 R 감쇠 큼)
    beta = np.array([random.uniform(0.8,1.5),  # B
                     random.uniform(1.0,2.0),  # G
                     random.uniform(1.8,3.0)], np.float32)  # R
    A = np.array(A, np.float32)
    I = bgr.astype(np.float32)
    out = np.zeros_like(I)
    for c in range(3):
        trans = np.exp(-beta[c] * d)  # T(x,y)
        out[...,c] = I[...,c]*trans + A[c]*(1.0-trans)
    return np.clip(out, 0, 255).astype(np.uint8)

def fda_lowfreq_swap(src, ref, L=0.02):
    # src/ref: HWC BGR uint8
    # L: 저주파 비율(0.01~0.05 권장)
    S = np.fft.fft2(cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32), axes=(0,1))
    R = np.fft.fft2(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32), axes=(0,1))
    S_amp, S_pha = np.abs(S), np.angle(S)
    R_amp = np.abs(R)
    h, w = src.shape[:2]
    rh, rw = int(h*L), int(w*L)
    cy, cx = h//2, w//2
    S_amp[cy-rh:cy+rh+1, cx-rw:cx+rw+1, :] = R_amp[cy-rh:cy+rh+1, cx-rw:cx+rw+1, :]
    S_new = S_amp * np.exp(1j * S_pha)
    out = np.real(np.fft.ifft2(S_new, axes=(0,1)))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ---- 콜백 구현 ----
def on_preprocess_batch(trainer):
    """
    Ultralytics 콜백: 학습 배치 직전 호출.
    trainer.batch: dict('img','cls','bboxes','masks',...)
    """
    if not trainer.training:
        return
    imgs_t = trainer.batch["img"]  # (B,3,H,W) float32[0,1] RGB (CUDA tensor)
    B = imgs_t.size(0)
    imgs = to_numpy(imgs_t)
    # masks: list of tensors -> list[list[np.ndarray(HW)]]
    raw_masks = trainer.batch.get("masks", None)
    masks = []
    if raw_masks is not None:
        for i in range(B):
            mi = raw_masks.data[i]  # (Ni,H,W) 0/1
            masks.append([mi[j].cpu().numpy().astype(np.uint8) for j in range(mi.shape[0])])
    else:
        masks = [[] for _ in range(B)]

    # 커리큘럼(초기 강, 말기 약)
    t = trainer.epoch / max(1, trainer.epochs)
    base = 0.35 * (0.6 + 0.4*(1.0 - t))  # 후반 약화

    for i, im in enumerate(imgs):
        # 순서: WB → 색캐스트 → 감마/대비 → 헤이즈 → 언샵 → CLAHE(확률)
        if random.random() < 0.9: im = gray_world_wb(im)
        if random.random() < 0.9: im = blue_green_cast(im, s=0.15*base)
        if random.random() < 0.9:
            g = random.uniform(0.7, 1.4* (1+0.3*base))
            c = random.uniform(1-0.4*base, 1+0.4*base)
            im = gamma_contrast(im, g=g, c=c)
        if random.random() < 0.7: im = add_haze(im, alpha=random.uniform(0, 0.35*base))
        if random.random() < 0.7: im = unsharp(im, k=random.uniform(0, 0.5*base))
        if random.random() < 0.3: im = lab_clahe(im, clip=2.0+1.5*base)
        # ① 공간적 헤이즈(확률 0.5)
        if random.random() < 0.5:
            a = perlin_like_mask(im.shape[0], im.shape[1], scale=0.04)
            A = np.array([210, 232, 255], np.float32)
            hazy = im.astype(np.float32)*(1-a[...,None]) + A*(a[...,None])
            im = np.clip(hazy,0,255).astype(np.uint8)

        # ② Beer–Lambert 색감쇠(확률 0.5)
        if random.random() < 0.5:
            im = beer_lambert_underwater(im)

        # ③ FDA 저주파 교환(확률 0.3, 같은 배치 다른 이미지와)
        if random.random() < 0.3 and len(imgs) > 1:
            j = random.randrange(len(imgs))
            if j != i:
                im = fda_lowfreq_swap(im, imgs[j], L=random.uniform(0.01, 0.03))
        imgs[i] = im

    imgs, masks = copy_paste_masks(imgs, masks, p=0.25*(0.8+0.2*(1.0-t)), max_paste=2)
    trainer.batch["img"] = to_tensor(imgs).to(imgs_t.device)
    # masks는 그대로 사용(복사-붙여넣기 효과가 GT에는 반영되지 않음 → 작은 객체 복잡도만 높여 일반화에 도움)
