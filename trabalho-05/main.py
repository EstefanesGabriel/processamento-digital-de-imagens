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

IMAGES = ["0.bmp", "1.bmp", "2.bmp", "3.bmp", "4.bmp", "5.bmp", "6.bmp", "7.bmp", "8.bmp"]
BACKGROUND = "astronaut.png"

#===============================================================================

def mask(img):
    hls = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HLS)
    lower_green = np.array([40, 50, 60])   # H, L, S
    upper_green = np.array([75, 200, 255])
    
    mask = cv2.inRange(hls, lower_green, upper_green)

    mask = np.where(mask[...] != 0, 1, 0)

    return mask

#-------------------------------------------------------------------------------

def apply_image(img, mask, bg_path):
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    if bg is None:
        print('Erro abrindo a imagem.\n')
        return
    bg = bg.reshape((bg.shape[0], bg.shape[1], bg.shape[2]))

    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    if bg.shape[2] == 4:
        bg = bg[:, :, :3]

    bg_removed = np.where(mask[..., None] == 0, img, 0)

    img_float = bg_removed.astype(np.float32)
    b, g, r = cv2.split(img_float)
    spill_mask = (g > r * 1.2) & (g > b * 1.2)

    g[spill_mask] *= 0.3

    corrected = cv2.merge([b, g, r]).astype(np.uint8)

    result = np.where(mask[..., None] == 0, corrected, bg)

    return result

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    for bmp in IMAGES:
        in_path = os.path.join("img", bmp)
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Erro abrindo a imagem.\n')
            break
        img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        
        binario = mask(img.copy())

        bg_path = os.path.join("background", BACKGROUND)
        resultado = apply_image(img.copy(), binario, bg_path)
        if resultado is None:
            break

        out_path = os.path.join("resultados", (bmp.split('.')[0]) + '.png')

        cv2.imwrite(out_path, resultado)
