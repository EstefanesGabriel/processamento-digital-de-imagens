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

if __name__ == "__main__":
    if not os.path.exists("resultados"):
        os.mkdir("resultados")
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Erro abrindo a imagem.\n")
            break
        img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        img = img.astype(np.float32) / 255

        mask = create_mask(img.copy())
        mask, fg = remove_green(img.copy(), mask)
        result = apply_image(fg, mask, BACKGROUND)

        out_path = os.path.join("resultados", str(i) + ".png")
        cv2.imwrite(out_path, (result * 255).astype(np.uint8)) # type: ignore
