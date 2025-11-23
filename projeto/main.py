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
    # 'dataset/0.png',
    # 'dataset/1.png',
    # 'dataset/2.png',
    # 'dataset/3.png',
    'dataset/4.png'
    # 'dataset/5.png',
    # 'dataset/6.png',
    # 'dataset/7.png',
    # 'dataset/8.png',
    # 'dataset/9.png',
]

#===============================================================================

def binary(img):
    img_blur = cv2.GaussianBlur(img, (0, 0), 100)

    img_subtract =  img - img_blur

    _, img_binary = cv2.threshold(img_subtract, 0.01, 1, cv2.THRESH_BINARY)

    img_binary = np.where(img_binary == 1, 0, 1).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_eroded = cv2.erode(img_binary, kernel, iterations=3)
    img_dilated = cv2.dilate(img_eroded, kernel, iterations=3)

    return img_dilated

def platelet_count(treated_img):
    contours, _ = cv2.findContours(treated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # areas = [cv2.contourArea(c) for c in contours]
    # areas.sort()
    #
    # # limite_superior = max(limite_inferior + 1, int(len(areas) * 0.4))
    # # median_areas = areas[limite_inferior:limite_superior]
    #
    # avg_rice_area = np.median(median_areas)
    #
    # desvio = np.std(areas)
    # media = np.mean(areas)
    #
    # if desvio / media < 0.25:  # tolerância ajustável
    #     return len(contours)
    #
    # total_rice_count = 0
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     count_in_contour = round(area / (avg_rice_area * FATOR_DE_CALIBRAGEM))
    #
    #     if count_in_contour == 0 and area > avg_rice_area * 0.5:
    #         count_in_contour = 1
    #
    #     total_rice_count += count_in_contour
    #
    # return total_rice_count

if __name__ == '__main__':
    if not os.path.exists("resultados"):
        os.mkdir("resultados")
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro abrindo a imagem: {IMAGES[i]}\n")
            continue

        img = img.astype (np.float32) / 255

        result = binary(img.copy())

        out_path = os.path.join("resultados", str(i) + ".png")
        cv2.imwrite(out_path, (result * 255).astype(np.uint8))
