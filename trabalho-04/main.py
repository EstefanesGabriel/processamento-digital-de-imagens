import cv2
import numpy as np
import os

# --- PARÂMETRO DE AJUSTE PARA MAIS ITERAÇÕES ---
# Aumente este fator para ter um processo de erosão mais lento e detalhado.
# 2 = imagem 2x maior (4x mais pixels).
# 3 = imagem 3x maior (9x mais pixels).
FATOR_ESCALA = 2

# --- PREPARAÇÃO DO DIRETÓRIO DE SAÍDA ---
output_folder = "iteracoes_erosao_escalada"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(f"As imagens de cada iteração serão salvas na pasta: '{output_folder}/'")

# --------------------------------------------------------------------------
# PARTE 1: LÓGICA ORIGINAL COM OTSU PARA CRIAR A MÁSCARA BASE
# --------------------------------------------------------------------------
try:
    image = cv2.imread('150.bmp')
    if image is None:
        raise FileNotFoundError("O arquivo de imagem '150.bmp' não foi encontrado ou não pôde ser lido.")
    print("Imagem '150.bmp' carregada.")
except FileNotFoundError as e:
    print(e)
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
(T, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Máscara binária inicial criada.")


# ---------------------------------------------------------------------------------
# PARTE 2: LÓGICA DE EROSÃO ITERATIVA COM IMAGEM REDIMENSIONADA
# ---------------------------------------------------------------------------------

# --- ETAPA DE REDIMENSIONAMENTO ---
# Aumentamos o tamanho da máscara para um processo de erosão mais suave.
altura, largura = thresh.shape
nova_altura = altura * FATOR_ESCALA
nova_largura = largura * FATOR_ESCALA
mascara_escalada = cv2.resize(thresh, (nova_largura, nova_altura), interpolation=cv2.INTER_NEAREST)
print(f"Máscara redimensionada por um fator de {FATOR_ESCALA}. Iniciando erosão iterativa...\n")

# Inicializa as variáveis
max_rice_count = 0
eroded_mask = mascara_escalada.copy()
kernel = np.ones((3,3), np.uint8)
iteration = 0

while True:
    iteration += 1
    eroded_mask = cv2.erode(eroded_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_count = len(contours)
    
    # --- LÓGICA DE VISUALIZAÇÃO ---
    frame_to_save = image.copy()
    
    # IMPORTANTE: Os contornos estão na escala aumentada. Precisamos convertê-los de volta
    # para a escala original antes de desenhar.
    if len(contours) > 0:
        contornos_originais = []
        for c in contours:
            # Converte o contorno para float, divide pelo fator de escala, e converte de volta para int
            c_orig = (c / FATOR_ESCALA).astype(np.int32)
            contornos_originais.append(c_orig)
        
        cv2.drawContours(frame_to_save, contornos_originais, -1, (0, 255, 0), 2)
    
    text = f"Iteracao: {iteration} | Contagem: {current_count}"
    cv2.putText(frame_to_save, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    filename = os.path.join(output_folder, f"iteracao_{iteration:03d}.png") # :03d para centenas de iterações
    cv2.imwrite(filename, frame_to_save)
    # --- FIM DA LÓGICA DE VISUALIZAÇÃO ---

    print(f"Iteração {iteration}: {current_count} grãos detectados.")

    if current_count > max_rice_count:
        max_rice_count = current_count

    if current_count == 0:
        break

print("\nProcesso de erosão concluído.")
print(f"--------------------------------------------------")
print(f"RESULTADO FINAL: A contagem máxima de grãos atingida foi de {max_rice_count}.")
print(f"(Alvo: 150)")
print(f"--------------------------------------------------")