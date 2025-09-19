#===============================================================================
# Projeto - Trabalho 3: Bloom.
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
# Nomes:
# Alexandre Alberto Menon - 2603403
# Gabriel Rodrigues Estefanes - 2603446
#===============================================================================

import numpy as np
import cv2
import sys

#===============================================================================

INPUT_IMAGE = './Wind Waker GC.bmp'
BRIGHT = 1.5
CONSTANT_GB = 0.3
CONSTANT_BB = 0.3

#===============================================================================

def bright_pass(img):
    rows = img.shape[0]
    cols = img.shape[1]

    for row in range(rows):
        for col in range(cols):
            brightness = 0
            brightness = img[row][col][0] + img[row][col][1] + img[row][col][2]
            if brightness < BRIGHT:
                img[row][col][:] = 0

    return img

#-------------------------------------------------------------------------------

def gaussian_blur(img):
    sigma = 1
    kernel = sigma * 7
    img_out = cv2.GaussianBlur(img, (kernel, kernel), sigma, cv2.BORDER_REFLECT)
    sigma += 1
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]

    while sigma < 20:
        kernel = sigma*7
        if kernel % 2 == 0:
            kernel -= 1
        img_out += cv2.GaussianBlur(img, (kernel, kernel), sigma, cv2.BORDER_REFLECT)
        sigma *= 2

    for row in range(rows):
        for col in range(cols):
            for ch in range(channels):
                if img_out[row][col][ch] > 1:
                    img_out[row][col][ch] = 1

    return img_out

#-------------------------------------------------------------------------------

def box_blur(img):
    kernel = 15
    img_out = cv2.blur(img, (kernel,kernel), cv2.BORDER_REFLECT)
    sigma = 2
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]

    while sigma < 20:
        img_aux = img.copy()
        for _ in range(int(sigma/2)):
            img_aux = cv2.blur(img_aux, (kernel, kernel), cv2.BORDER_REFLECT)
        img_out += img_aux
        sigma *= 2

    for row in range(rows):
        for col in range(cols):
            for ch in range(channels):
                if img_out[row][col][ch] > 1:
                    img_out[row][col][ch] = 1
    
    return img_out

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype(np.float32) / 255

    img_bp = bright_pass(img.copy())
    img_gb = gaussian_blur(img_bp.copy())
    img_bb = box_blur(img_bp.copy())

    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]

    img1 = img + img_gb * CONSTANT_GB
    img2 = img + img_bb * CONSTANT_BB

    for row in range(rows):
        for col in range(cols):
            for ch in range(channels):
                if img1[row][col][ch] > 1:
                    img1[row][col][ch] = 1
                if img2[row][col][ch] > 1:
                    img2[row][col][ch] = 1

    cv2.imwrite('bloomgb-image.png', img1*255)
    # cv2.imshow('bloomgb-image', (img1 * 255).astype(np.uint8))
    cv2.imwrite('bloombb-image.png', img2*255)
    # cv2.imshow('bloombb-image', (img2 * 255).astype(np.uint8))
    # cv2.imwrite('bright-image.png', img_bp*255)
    # cv2.imshow('bright-image', (img_bp * 255).astype(np.uint8))
    # cv2.imwrite('gaussian-image.png', img_gb*255)
    # cv2.imshow('gaussian-image', (img_gb * 255).astype(np.uint8))
    # cv2.imwrite('boxblur-image.png', img_bb*255)
    # cv2.imshow('boxblur-image', (img_bb * 255).astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
