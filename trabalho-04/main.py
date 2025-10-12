#===============================================================================
# Projeto - Trabalho 4: Contagem de arroz.
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
# Nomes:
# Alexandre Alberto Menon - 2603403
# Gabriel Rodrigues Estefanes - 2603446
#===============================================================================

import cv2
import numpy as np

#===============================================================================

FATOR_DE_CALIBRAGEM = 1.042
IMAGES = ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']

#===============================================================================

def image_treatment (img):
    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img_gray = img_gray.astype(np.float32) / 255

    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 20)

    subtract_img = img_gray - img_blur

    _, img_thresh = cv2.threshold(subtract_img, 0.165, 1, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_eroded = cv2.erode(img_thresh, kernel, iterations=2)
    img_dilated = cv2.dilate(img_eroded, kernel, iterations=2)

    treated_img = (img_dilated * 255).astype(np.uint8)

    return treated_img

#-------------------------------------------------------------------------------

def rices_contour(treated_img):
    contours, _ = cv2.findContours(treated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    areas.sort()
    limite_inferior = max(0, int(len(areas) * 0.1))
    limite_superior = max(limite_inferior + 1, int(len(areas) * 0.4))
    median_areas = areas[limite_inferior:limite_superior]

    avg_rice_area = np.median(median_areas)

    desvio = np.std(areas)
    media = np.mean(areas)

    if desvio / media < 0.25:  # tolerância ajustável
        return len(contours)

    total_rice_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        count_in_contour = round(area / (avg_rice_area * FATOR_DE_CALIBRAGEM))
        
        if count_in_contour == 0 and area > avg_rice_area * 0.5:
            count_in_contour = 1
            
        total_rice_count += count_in_contour

    return total_rice_count

#-------------------------------------------------------------------------------

def print_total_rices(total_rices):
    print(f"\n--------------------------------------------------")
    print(f"RESULTADO: {int(total_rices)} grãos de arroz contados.")
    print(f"--------------------------------------------------")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    for img in IMAGES:
        treated_img = image_treatment(img)
        total_rices = rices_contour(treated_img)
        print_total_rices(total_rices)
