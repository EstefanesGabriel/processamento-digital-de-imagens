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
]

#===============================================================================
def remove_bg(img):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 100)
    img_subtract =  img_gray - img_blur

    _, binary = cv2.threshold(img_subtract, 0.01, 1, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    eroded = cv2.erode(binary, kernel, iterations=3)
    mask = cv2.dilate(eroded, kernel, iterations=5)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    fg = np.where(mask == 1, img, 0).astype(np.float32)

    return fg

def filter_cells(img):
    pixel_val = lambda pix, r1, s1, r2, s2: ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1

    r1 = 70
    r2 = 100
    s1 = 0
    s2 = 255

    pixel_val_vec = np.vectorize(pixel_val)

    filtered = pixel_val_vec(img, r1, s1, r2, s2)

    binary = np.where(filtered >= 125, 255, 0).astype(np.uint8)

    return binary

def k_means(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    aux = hsv[:, :, :2]
    z = aux.reshape((-1, 2))
    z = np.float32(z)
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    _, labels, _ = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # pyright: ignore

    labels_flat = labels.flatten()

    palette = np.array([
        [255, 0, 0],      # cluster 0
        [0, 255, 0],      # cluster 1
        [0, 0, 255],      # cluster 2
        [255, 255, 0]     # cluster 3
    ], dtype=np.uint8)

    colored = palette[labels_flat]
    colored = colored.reshape(img.shape)

    return colored

if __name__ == '__main__':
    if not os.path.exists("resultados"):
        os.mkdir("resultados")
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Erro abrindo a imagem: {IMAGES[i]}\n")
            continue

        img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        img = img.astype(np.float32) / 255

        fg = remove_bg(img.copy())

        fg = np.clip(fg * 255, 0, 255).astype(np.uint8)
        filtered = filter_cells(fg.copy())
        out = k_means(filtered)

        b, g, _ = cv2.split(out)
        white = np.count_nonzero(b == 255)
        black = np.count_nonzero(b == 0)

        if black > white:
            mask = b
        else:
            mask = g

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img_dilated = cv2.dilate(mask, kernel, iterations=15)
        img_eroded = cv2.erode(img_dilated, kernel, iterations=25)
        img_white_cell = cv2.dilate(img_eroded, kernel, iterations=25)
        img_platelet = mask - img_white_cell

        img_platelet = cv2.dilate(img_platelet, kernel, iterations=15)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        filtered = np.where(filtered >= 25, 255, 0).astype(np.uint8)

        filtered -= img_platelet
        img_white_cell = cv2.dilate(img_white_cell, kernel, iterations=15)
        filtered = np.where(img_white_cell[...] != 0, 0, filtered)

        cv2.imshow('', filtered)
        cv2.waitKey()
        cv2.destroyAllWindows()
