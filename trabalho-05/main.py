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
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    mask = (g > 0.3) & ((g - np.maximum(r, b)) > 0.1)

    binario = np.where(mask, 0, 1).astype(np.float32)

    return binario

#-------------------------------------------------------------------------------

def apply_image(img, mask, bg_path):
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    if bg is None:
        print('Erro abrindo a imagem.\n')
        return
    bg = bg.reshape((bg.shape[0], bg.shape[1], bg.shape[2]))
    bg = bg.astype(np.float32) / 255

    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    if bg.shape[2] == 4:
        bg = bg[:, :, :3]

    resultado = np.where(mask[..., None] == 0, bg, img)

    return resultado

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    for bmp in IMAGES:
        in_path = os.path.join("img", bmp)
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Erro abrindo a imagem.\n')
            break
        img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        img = img.astype(np.float32) / 255
        
        binario = mask(img.copy())

        bg_path = os.path.join("background", BACKGROUND)
        resultado = apply_image(img.copy(), binario, bg_path)
        if resultado is None:
            break

        out_path = os.path.join("resultados", (bmp.split('.')[0]) + '.png')

        cv2.imwrite(out_path, (resultado * 255).astype(np.uint8))
