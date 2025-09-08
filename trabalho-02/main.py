#===============================================================================
# Projeto - Trabalho 2: Blur.
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
# Nomes:
# Alexandre Alberto Menon - 2603403
# Gabriel Rodrigues Estefanes - 2603446
#===============================================================================

import sys
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  './Exemplos/b01 - Original.bmp'
W = 11
H = 15

#===============================================================================

def blur_ingenuo(img, w, h):
    lim_w = int((w-1)/2)
    lim_h = int((h-1)/2)
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]
    blur = img.copy()

    for row in range(lim_h, rows-lim_h):
        for col in range(lim_w, cols-lim_w):
            for ch in range(channels):
                sum = 0
                for i in range(row-lim_h, row+lim_h+1):
                    for j in range(col-lim_w, col+lim_w+1):
                        sum = sum + img[i][j][ch]
                        
                blur[row][col][ch] = sum / (w*h)

    return blur

#-------------------------------------------------------------------------------

def blur_separavel(img, w, h):
    lim_w = int((w-1)/2)
    lim_h = int((h-1)/2)
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]
    aux = img.copy()
    blur = img.copy()

    for row in range(lim_h, rows-lim_h):
        for col in range(lim_w, cols-lim_w):
            for ch in range(channels):
                sum = 0
                for i in range(col-lim_w, col+lim_w+1):
                    sum = sum + img[row][i][ch]

                aux[row][col][ch] = sum / w

    for row in range(lim_h, rows-lim_h):
        for col in range(lim_w, cols-lim_w):
            for ch in range(channels):
                sum = 0
                for i in range(row-lim_h, row+lim_h+1):
                    sum = sum + aux[i][col][ch]

                blur[row][col][ch] = sum / h

    return blur

#-------------------------------------------------------------------------------

def blur_integral(img, w, h):
    lim_w = int((w-1)/2)
    lim_h = int((h-1)/2)
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]
    integral = integralizacao(img.copy())
    blur = img.copy()

    for row in range(rows):
        for col in range(cols):
            if row-lim_h-1 < 0: # lim+1 > row
                row_minus = 0
                v_up = row-1
            else:
                row_minus = row-lim_h-1
                v_up = lim_h
            if row+lim_h > rows-1:
                row_plus = rows-1
                v_down = (rows-1)-row
            else: 
                row_plus = row+lim_h
                v_down = lim_h
            if col-lim_w-1 < 0:
                col_minus = 0
                h_left = col-1
            else:
                col_minus = col-lim_w-1
                h_left = lim_w
            if col+lim_w > cols-1:
                col_plus = cols-1
                h_right = (cols-1)-col
            else:
                col_plus = col+lim_w
                h_right = lim_w

            for ch in range(channels):
                blur[row][col][ch] = (
                    integral[row_plus][col_plus][ch]
                    - integral[row_plus][col_minus][ch]
                    - integral[row_minus][col_plus][ch]
                    + integral[row_minus][col_minus][ch]
                ) / ((v_up+v_down+1)*(h_left+h_right+1))

    return blur

#-------------------------------------------------------------------------------

def integralizacao(img):
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]
    for row in range(rows):
        for col in range(cols):
            for ch in range(channels):
                if row == 0 and col == 0:
                    pass
                else:
                    if row == 0:
                        img[row][col][ch] = img[row][col][ch] + img[row][col-1][ch]
                    elif col == 0:
                        img[row][col][ch] = img[row][col][ch] + img[row-1][col][ch]
                    else:
                        img[row][col][ch] = (
                                img[row][col][ch]
                                + img[row][col-1][ch]
                                + img[row-1][col][ch]
                                - img[row-1][col-1][ch]
                        )
    return img

#===============================================================================

if __name__ == '__main__':
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255

    img_cv2_blur = cv2.blur(img, (W, H))
    img_ingenuo = blur_ingenuo(img, W, H)
    img_separavel = blur_separavel(img, W, H)
    img_integral = blur_integral(img, W, H)
    cv2.imwrite ('01-ingenuo.png', img_ingenuo*255)
    cv2.imwrite ('cv2-blur.png', img_cv2_blur*255)
    cv2.imwrite ('01-separavel.png', img_separavel*255)
    cv2.imwrite ('01-integral.png', img_integral*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()
