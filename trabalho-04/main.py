#===============================================================================
# Projeto - Trabalho 4: Contagem de arroz.
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

#===============================================================================

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype(np.float32) / 255

