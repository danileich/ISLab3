# -*- coding: utf-8 -*-
from copy import deepcopy
from PIL import Image
import numpy as np
from math import sqrt

def PixelAnalysis(imageA, w, h):
    p = []
    pixels = imageA.load()
    for a in range(h):
        p.append([])
        for b in range(w):
            if pixels[b, a][0] < 128:
                p[-1].append(1)
            else:
                p[-1].append(0)
    return p

def TransformationMatrix(m):
    temporary = []
    for a in range(14):
        for b in range(12):
            temporary.append(m[a][b])
    return temporary

def PrintMassive(massive, w, h):
    global txtLine
    txt = open('matrix-paint.txt', 'r+', encoding="utf-8")
    txt.read(txtLine+1)
    for i in range(h):
        for j in range(w):
            if massive[i][j] == 1:
                txt.write(u'■ ')
            else:
                txt.write(u'□ ')
        txt.write('\n')
    txt.write('\n')
    txt.close()
    txtLine += h + 1

def MatrixCore(m):
    mCore = []
    for a in range(12):
        amount = 0
        for b in m:
            amount += b
        amount /= 14
        mCore.append(amount)
    return mCore

def CovarianceMatrix(m):
    matrix = deepcopy(m)
    matrix = np.cov(matrix)
    return matrix

def TranscendenceMatrix(m):
    matrix = []
    h, w = len(m), len(m[0])
    for b in range(w):
        matrix.append([])
        for a in range(h):
            matrix.append(m[a][b])
    return matrix

def MatrixMultiplication(m1, m2):
    matrix = []
    h1, h2, w1, w2 = len(m1), len(m2), len(m1[0]), len(m2[0])
    if w1 != h2:
        return
    for a in range(h1):
        matrix.append([])
        for b in range(w2):
            amount = 0
            for c in range(w1):
                amount += m1[a][c] * m2[c][b]
            matrix[-1].append(amount)
    return matrix

def InverseMatrix(m):
    return np.linalg.inv(m)

def EuclideanDistance(x, y):
    distance = sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))
    return distance

def MahalanobisEuclidDistance(matrix, c):
    core = c
    inverse = InverseMatrix(matrix)
    for i in range(len(matrix[0])):
        matrix[0][i] = matrix[0][i] - core[i]
    trM = TranscendenceMatrix(matrix)
    amount1 = MatrixMultiplication(matrix, inverse)
    amount2 = MatrixMultiplication(amount1, trM)
    distance = amount2[0][0]
    distance = sqrt(distance)
    return distance

def ClassSelection(distM, letter):
    global classT, classZ, classG, classN
    if distM[0] < 2:
        classT.append(letter)
    if distM[1] < 7:
        classZ.append(letter)
    if distM[2] < 14:
        classG.append(letter)
    if distM[3] < 21:
        classN.append(letter)
    return


txtLine = 0
N0 = Image.open('images/N(0).png')
N1 = Image.open('images/N(1).png')
N2 = Image.open('images/N(2).png')
K3 = Image.open('images/K(3).png')
K4 = Image.open('images/K(4).png')
K5 = Image.open('images/K(5).png')
T6 = Image.open('images/T(6).png')
T7 = Image.open('images/T(7).png')
T8 = Image.open('images/T(8).png')
Z9 = Image.open('images/Z(9).png')
Z10 = Image.open('images/Z(10).png')
Z11 = Image.open('images/Z(11).png')
matrixList = []
classT, classZ, classG, classN = [], [], [], []
width, height = N0.size[0], N0.size[1]
for p in (N0, N1, N2, K3, K4, K5, T6, T7, T8, Z9, Z10, Z11):
    matrixList.append(PixelAnalysis(p, width, height))
classT.append(MatrixCore(N0))
classZ.append(MatrixCore(K3))
classG.append(MatrixCore(T6))
classN.append(MatrixCore(Z9))
metrica = []
for image in matrixList:
    for ob in (classT[0], classZ[0], classG[0], classN[0]):
        imageLine = TransformationMatrix(image)
        distME = MahalanobisEuclidDistance(imageLine, ob)
        metrica.append(distME)
    ClassSelection(metrica, image)
    metrica = []
