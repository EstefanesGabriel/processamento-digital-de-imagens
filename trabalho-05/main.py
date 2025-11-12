#===============================================================================
# Projeto - Trabalho 5: Chroma Key.
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
# Nomes:
# Alexandre Alberto Menon - 2603403
# Gabriel Rodrigues Estefanes - 2603446
#===============================================================================

import numpy as np
import cv2
import os

#===============================================================================

IMAGES = [
    "img/0.bmp", 
    "img/1.bmp", 
    "img/2.bmp", 
    "img/3.bmp", 
    "img/4.bmp", 
    "img/5.bmp", 
    "img/6.bmp", 
    "img/7.bmp", 
    "img/8.bmp"
]

BACKGROUND = "background/astronaut.png"

#===============================================================================

def create_mask(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    green = 1 + np.maximum(b,r) - g
    green = np.clip(green, 0.0, 1.0)

    return green

#-------------------------------------------------------------------------------

def remove_green(img, mask):
    alpha = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX) # type: ignore
    alpha = np.clip(alpha * 1.5 - 0.5, 0, 1)

    removed = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    removed[:, :, 2] *= alpha
    removed = cv2.cvtColor(removed, cv2.COLOR_HLS2BGR)

    return alpha, removed

#-------------------------------------------------------------------------------

def apply_image(fg, mask, bg_path):
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    if bg is None:
        print("Erro abrindo a imagem de background.\n")
        return
    bg = bg.reshape((bg.shape[0], bg.shape[1], bg.shape[2]))
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
    if bg.shape[2] == 4:
        bg = bg[:, :, :3]
    bg = bg.astype(np.float32) / 255

    result = fg * mask[:, :, None] + (bg * (1 - mask[:, :, None]))

    return result

#-------------------------------------------------------------------------------

def post_processing(img):
    if img is None:
        return img

    img = img.copy()
    if img.ndim != 3 or img.shape[2] < 3:
        return img

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hls = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HLS) 

    h = hls[:, :, 0].astype(np.int16)
    l = hls[:, :, 1].astype(np.float32)
    s = hls[:, :, 2].astype(np.float32)

    h_low, h_high = 35, 85    
    s_min = 10               
    l_min = 10               

    mask = (h >= h_low) & (h <= h_high) & (s >= s_min) & (l >= l_min)

    
    s[mask] = s[mask] * 0  

    hls_mod = np.stack([h.astype(np.uint8), np.clip(l, 0, 255).astype(np.uint8), np.clip(s, 0, 255).astype(np.uint8)], axis=2)
    bgr_mod_u8 = cv2.cvtColor(hls_mod, cv2.COLOR_HLS2BGR)
    bgr_mod = bgr_mod_u8.astype(np.float32) / 255.0

    return bgr_mod

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists("resultados"):
        os.mkdir("resultados")
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Erro abrindo a imagem: {IMAGES[i]}\n")
            continue

        # garantir 3 canais válidos
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.astype(np.float32) / 255

        mask = create_mask(img.copy())
        alpha, fg = remove_green(img.copy(), mask)

        applied = apply_image(fg, alpha, BACKGROUND)
        if applied is None:
            print("Erro aplicando background. Pulando imagem.\n")
            continue
        result = applied  # desempacota o resultado correto

        img_chr = post_processing(result)

        out_path = os.path.join("resultados", str(i) + ".png")
        cv2.imwrite(out_path, (img_chr * 255).astype(np.uint8))
