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
W = 3

#===============================================================================

def blur(img, w):
    lim = int((w-1)/2)
    rows =(img.shape[0])
    cols = img.shape[1]
    blur = img

    for row in range(lim, rows-lim):
        for col in range(lim, cols-lim):
            sum = 0
            for i in range(row-lim, row+lim):
                for j in range(col-lim, col+lim):
                    sum = sum + img[i][j]
                    
            blur[row][col] = sum / (w**2)

    return blur

#===============================================================================

if __name__ == '__main__':
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    img_ingenuo = blur (img, W)
    cv2.imshow ('01 - ingenuo', img)
    cv2.imwrite ('01 - ingenuo.png', img*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()
