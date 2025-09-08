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
W = 11

#===============================================================================

def blur_ingenuo(img, w):
    lim = int((w-1)/2)
    rows = img.shape[0]
    cols = img.shape[1]
    blur = img.copy()

    for row in range(lim, rows-lim):
        for col in range(lim, cols-lim):
            sum = 0
            for i in range(row-lim, row+lim+1):
                for j in range(col-lim, col+lim+1):
                    sum = sum + img[i][j]
                    
            blur[row][col] = sum / (w**2)

    return blur

def blur_separavel(img, w):
    lim = int((w-1)/2)
    rows = img.shape[0]
    cols = img.shape[1]
    aux = img.copy()
    blur = img.copy()

    for row in range(lim, rows-lim):
        for col in range(lim, cols-lim):
            sum = 0
            for i in range(col-lim, col+lim+1):
                sum = sum + img[row][i]

            aux[row][col] = sum / w

    for row in range(lim, rows-lim):
        for col in range(lim, cols-lim):
            sum = 0
            for i in range(row-lim, row+lim+1):
                sum = sum + aux[i][col]

            blur[row][col] = sum / w

    return blur

def blur_integral(img, w):
    lim = int((w-1)/2)
    rows = img.shape[0]
    cols = img.shape[1]
    integral = integralizacao(img.copy())
    blur = img.copy()

    for row in range(rows):
        for col in range(cols):
            if col - lim - 1 < 0 and row - lim - 1 < 0:
                blur[row][col] = (
                    integral[row+lim][col+lim] 
                    - integral[row+lim][0] 
                    - integral[0][col+lim] 
                    + integral[0][0]
                ) / (w**2)
            else:
                if col - lim - 1 < 0:
                    blur[row][col] = (
                        integral[row+lim][col+lim] 
                        - integral[row+lim][0] 
                        - integral[row-lim-1][col+lim] 
                        + integral[row-lim-1][0]
                    ) / (w**2)
                elif row - lim - 1 < 0:
                    blur[row][col] = (
                        integral[row+lim][col+lim] 
                        - integral[row+lim][col-lim-1] 
                        - integral[0][col+lim] 
                        + integral[0][col-lim-1]
                    ) / (w**2)
                else:
                    blur[row][col] = (
                        integral[row+lim][col+lim] 
                        - integral[row+lim][col-lim-1] 
                        - integral[row-lim-1][col+lim] 
                        + integral[row-lim-1][col-lim-1]
                    ) / (w**2)
    return blur


def integralizacao(img):
    rows = img.shape[0]
    cols = img.shape[1]
    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:
                img[row][col] = img[row][col]
            else:
                if row == 0:
                    img[row][col] = img[row][col] + img[row][col-1]
                elif col == 0:
                    img[row][col] = img[row][col] + img[row-1][col]
                else:
                    img[row][col] = img[row][col] + img[row-1][col-1]
    return img



#===============================================================================

if __name__ == '__main__':
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    img_cv2_blur = cv2.blur(img, (11, 11))
    img_ingenuo = blur_ingenuo(img, W)
    img_separavel = blur_separavel(img, W)
    img_integral = blur_integral(img, W)
    cv2.imwrite ('01-ingenuo.png', img_ingenuo*255)
    cv2.imwrite ('cv2-blur.png', img_cv2_blur*255)
    cv2.imwrite ('01-separavel.png', img_separavel*255)
    cv2.imwrite ('01-integral.png', img_integral*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()
