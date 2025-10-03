import cv2
import numpy as np

# --- PARÂMETRO PRINCIPAL DE AJUSTE ---
# Se a contagem final está muito ALTA, AUMENTE este fator (ex: 1.05, 1.1, 1.15).
# Se a contagem final está muito BAIXA, DIMINUA este fator (ex: 0.95, 0.9).
FATOR_DE_CALIBRAGEM = 1.13 # Comece com 1.05 para uma pequena redução na contagem.
TOTAL = 205
INPUT_IMAGE = f'{TOTAL}.bmp'
C = -10
BLOCK_SIZE = 21

img_gray = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
cv2.imshow('img_gray', img_gray)

img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0)
# cv2.imshow('img_blur', img_blur)

(T, img_thresh) = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img_thresh_adapt = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)
# cv2.imwrite('img_thresh_adapt.bmp', img_thresh_adapt)
# cv2.imshow('img_thresh_adapt',img_thresh_adapt)

kernel_morph = np.ones((3,3), np.uint8)
img_eroded = cv2.erode(img_thresh_adapt, kernel_morph, iterations=2)
img_cleaned = cv2.dilate(img_eroded, kernel_morph, iterations=2)
cv2.imshow('img_cleaned', img_cleaned)


contours, _ = cv2.findContours(img_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_out = cv2.drawContours(img_gray.copy(), contours, -1, (0, 255, 0), 2)

print(contours[0])

# 3. Calcular a área média de um grão de arroz individual
areas = [cv2.contourArea(c) for c in contours]
areas.sort()
limite_inferior = max(0, int(len(areas) * 0.1))
limite_superior = max(limite_inferior + 1, int(len(areas) * 0.4))
median_areas = areas[limite_inferior:limite_superior]

if len(median_areas) == 0:
    print("Erro: Não há áreas suficientes para calcular a mediana.")
    exit(1)

avg_rice_area = np.median(median_areas)

# 4. Iterar sobre todos os contornos e estimar a contagem usando o fator de calibragem
total_rice_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area < avg_rice_area * 0.5:
        continue
    
    # --- MUDANÇA PRINCIPAL AQUI ---
    # Usamos a área média ajustada pelo fator de calibragem para a divisão.
    count_in_contour = round(area / (avg_rice_area * FATOR_DE_CALIBRAGEM))
    
    if count_in_contour == 0 and area > avg_rice_area * 0.5:
        count_in_contour = 1
        
    total_rice_count += count_in_contour

print(f"\n--------------------------------------------------")
print(f"RESULTADO CALIBRADO: {int(total_rice_count)} grãos de arroz contados.")
print(f"(Alvo: {TOTAL})")
print(f"--------------------------------------------------")

cv2.waitKey()
cv2.destroyAllWindows()