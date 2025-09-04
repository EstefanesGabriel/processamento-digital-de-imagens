#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
# Nomes:
# Alexandre Alberto Menon - 2603403
# Gabriel Rodrigues Estefanes - 2603446
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  './Exemplos/a01 - Original.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.6
ALTURA_MIN = 15
LARGURA_MIN = 15
N_PIXELS_MIN = 40

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    img_in = np.where(img > threshold, -1, 0).astype(np.float32)
    return img_in

#-------------------------------------------------------------------------------

def flood_fill(img, row, col, label, blob: list, i):
    img[row, col] = label
    blob[i]['n_pixels'] += 1

    if row > 0 and img[row-1, col] == -1:
        blob[i]['T'] = min(blob[i]['T'], row)
        flood_fill(img, row-1, col, label, blob, i)

    if row < img.shape[0] - 1 and img[row+1, col] == -1:
        blob[i]['B'] = max(blob[i]['B'], row)
        flood_fill(img, row+1, col, label, blob, i)

    if col > 0 and img[row, col-1] == -1:
        blob[i]['L'] = min(blob[i]['L'], col)
        flood_fill(img, row, col-1, label, blob, i)

    if col < img.shape[1] - 1 and img[row, col+1] == -1:
        blob[i]['R'] = max(blob[i]['R'], col)
        flood_fill(img, row, col+1, label, blob, i)

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
resiectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

    rows = img.shape[0]
    cols = img.shape[1]
    label = 0.1
    i = 0
    blob = []

    for row in range(rows):
        for col in range(cols):
            if img[row, col] == -1:
                blob.append({'label': label, 'n_pixels': 0, 'T': row, 'L': col, 'B': row, 'R': col})
                flood_fill(img, row, col, label, blob, i)
                if blob[i]['n_pixels'] < n_pixels_min or blob[i]['B'] - blob[i]['T'] < altura_min or blob[i]['R'] - blob[i]['L'] < largura_min:
                    blob.pop()
                    label -= 0.1
                    i -= 1
                label += 0.1
                i += 1
    return blob

#===============================================================================

def blur(img, w):
    rows =(img.shape[0])
    cols = img.shape[1]
    aux = img

    for row in range(w/2, rows-w/2):
        for col in range(w/2, cols-w/2):
            sum = 0
            for i in range(row-w/2, row+w/2):
                for j in range(col-w/2, col+w/2):
                    sum = sum + img[i][j]
                    
            aux[row][col] = sum / (w**2)

    return aux

#===============================================================================

if __name__ == '__main__':
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()
