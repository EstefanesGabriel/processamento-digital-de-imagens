#===============================================================================
# Projeto - Hemograma.
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
    'dataset/0.png',
    'dataset/1.png',
    'dataset/2.png',
    'dataset/3.png',
    'dataset/4.png',
    'dataset/5.png',
    'dataset/6.png',
    'dataset/7.png',
    'dataset/8.png',
]

#===============================================================================
def filter_white_cell(img):
    pixel_val = lambda pix, r1, s1, r2, s2: ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1

    r1 = 70
    s1 = 0
    r2 = 100
    s2 = 255

    pixel_val_vec = np.vectorize(pixel_val)

    filtered = pixel_val_vec(img, r1, s1, r2, s2)

    img_binary = np.where(filtered >= 125, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_dilated = cv2.dilate(img_binary, kernel, iterations=15)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=15)

    return img_eroded


# def binary(img):
#     img_blur = cv2.GaussianBlur(img, (0, 0), 100)
#
#     img_subtract =  img - img_blur
#
#     _, img_binary = cv2.threshold(img_subtract, 0.01, 1, cv2.THRESH_BINARY)
#
#     img_binary = np.where(img_binary == 1, 0, 1).astype(np.float32)
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     img_eroded = cv2.erode(img_binary, kernel, iterations=3)
#     img_dilated = cv2.dilate(img_eroded, kernel, iterations=3)
#
#     return img_dilated

def blob_count(treated_img):
    contours, _ = cv2.findContours(treated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    areas.sort(reverse=True)

    white = -1
    
    for i in range(len(areas) - 1):
        if areas[i] > areas[i+1] * 4:
            white = i + 1
            break

    platelet = len(areas) - white
    return white, platelet

if __name__ == '__main__':
    if not os.path.exists("resultados"):
        os.mkdir("resultados")
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro abrindo a imagem: {IMAGES[i]}\n")
            continue

        binary = filter_white_cell(img.copy())
        white, platelet = blob_count(binary.copy())

        print("=============")
        print(f"Imagem {i}")
        print("=============")
        print(f"Células brancas: {white}")
        print(f"Células platelet: {platelet}")
        out_path = os.path.join("resultados", str(i) + ".png")
        cv2.imwrite(out_path, binary)
