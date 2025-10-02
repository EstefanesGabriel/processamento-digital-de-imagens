import cv2
import numpy as np

# --- PARÂMETRO PRINCIPAL DE AJUSTE ---
# Se a contagem final está muito ALTA, AUMENTE este fator (ex: 1.05, 1.1, 1.15).
# Se a contagem final está muito BAIXA, DIMINUA este fator (ex: 0.95, 0.9).
FATOR_DE_CALIBRAGEM = 1.17 # Comece com 1.05 para uma pequena redução na contagem.


# 1. Carregar a imagem e criar a máscara binária
try:
    image = cv2.imread('150.bmp')
    if image is None: raise FileNotFoundError("Imagem não encontrada.")
except Exception as e:
    print(e)
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
(T, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. Encontrar todos os contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. Calcular a área média de um grão de arroz individual
areas = [cv2.contourArea(c) for c in contours]
areas.sort()
limite_inferior = int(len(areas) * 0.1)
limite_superior = int(len(areas) * 0.4)
median_areas = areas[limite_inferior:limite_superior]

if len(median_areas) == 0:
    print("Não foi possível estimar a área de um grão. Tente ajustar os limites.")
    exit()

avg_rice_area = np.median(median_areas)
print(f"Área média de um grão estimada: {avg_rice_area:.2f} pixels")
print(f"Área ajustada com fator de calibragem ({FATOR_DE_CALIBRAGEM}): {(avg_rice_area * FATOR_DE_CALIBRAGEM):.2f} pixels")


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
print(f"(Alvo: 150)")
print(f"--------------------------------------------------")