import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

##№ -------------- фильтр ------------------------###

NCoef = 2
DCgain = 32

ACoef = [112, 0, -112]
BCoef = [64, -117, 57]


def filter(x):
    y = np.zeros_like(x)

    for i in range(NCoef, x.shape[0]):
        y[i] = ACoef[0] * x[i]
        for j in range(1, NCoef + 1):
            y[i] += ACoef[j] * x[i - j] - BCoef[j] * y[i - j]

        y[i] /= BCoef[0]

    return y / DCgain

def aug_signal(signal):
    augmented_signal = np.zeros(signal.shape[0] * 2)
    for i in range(signal.shape[0]):
        augmented_signal[i] = signal[i]

    for i in range(signal.shape[0], augmented_signal.shape[0]):
        augmented_signal[i] = augmented_signal[i - 24]
    return augmented_signal


### -------------- отклонение от соседей ---------- ###

def sub_deviance(obj, i, j, k, l, rank = 1):

    value = 0
    denominator = 0 

    for a in range(-rank, 1 + rank, rank * 2):
        for b in range(-rank, 1 + rank, rank * 2):
            for c in range(-rank, 1 + rank, rank * 2):
                for d in range(-rank, 1 + rank, rank * 2):

                    try:
                        part = obj[i + a, j + b, k + c, d + l]

                        if part == 0:
                            continue

                        denominator += 1
                        value += np.abs(obj[i, j, k, l] - part)

                    except IndexError:
                        pass

    try:
        return value / denominator

    except ZeroDivisionError:
        return value


def deviance(obj, MammographMatrix, arguments = None, rank = 1):

    k, l, act = arguments
    
    new_x = np.zeros((18, 18))
    matrix = MammographMatrix

    for i in range(18):
        for j in range(18):

            if matrix[i, j] == -1:
                continue
            if arguments:
                if act == 'l':
                    new_x[i, j] = sub_deviance(obj, k, l, i, j, rank)
                else: 
                    new_x[i, j] = sub_deviance(obj, i, j, k, l, rank)
            else:    
                new_x[i, j] = sub_deviance(obj, i, j, i, j, rank)

    return new_x


### -------------- matrix voltage error ---------- ###

def matrix_voltage_error(x, MammographMatrix):

    matrix = MammographMatrix
    result = np.zeros((18, 18))

    for i in range(18):
        for j in range(18):
            for k in range(18):
                for l in range(18):

                    if matrix[i, j] == -1 or matrix[k, l] == -1:
                        continue
                    
                    h = 30

                    if abs(i - k) > h or abs(j - l) > h:
                        continue

                    result[i, j] += np.abs(x[i, j, k, l] - x[k, l, i, j])
    return result

### -------------- среднее по соседям ---------- ###

def sub_meas(obj, x_in, y_in, x_out, y_out, rank = 1):

    result = 0
    z = 0

    for a in range(-rank, 1 + rank, rank * 2):
        for b in range(-rank, 1 + rank, rank * 2):
            for c in range(-rank, 1 + rank, rank * 2):
                for d in range(-rank, 1 + rank, rank * 2):

                  try:
                    part = obj[x_in + a, y_in + b, x_out + c, y_out + d] 

                    if part == 0:
                        continue

                    result += part
                    z += 1

                  except IndexError:
                    pass

    try:
        return result / z

    except ZeroDivisionError:
        return result



def meas (obj, MammographMatrix, arguments = None, rank = 1):

    k, l, act = arguments

    new_x = np.zeros((18, 18))
    matrix = MammographMatrix

    for i in range(18):
        for j in range(18):

            if matrix[i, j] == -1:
                continue

            if arguments:
                if act == 'l':
                    new_x[i, j] = sub_meas(obj, k, l, i, j, rank)
                else: 
                    new_x[i, j] = sub_meas(obj, i, j, k, l, rank)
            else:    
                new_x[i, j] = sub_meas(obj, i, j, i, j, rank)

    return new_x

def sub_distance_meas(obj, x_in, y_in, x_out, y_out, rank = 1):

    result = 0
    z = 0

    mean_value = sub_meas(obj, rank, x_in, y_in, x_out, y_out)

    for a in range(-rank, 1 + rank, rank * 2):
        for b in range(-rank, 1 + rank, rank * 2):
            for c in range(-rank, 1 + rank, rank * 2):
                for d in range(-rank, 1 + rank, rank * 2):

                  try:
                    part = (mean_value - obj[x_in + a, y_in + b, x_out + c, y_out + d]) ** 2

                    if part == 0:
                        continue

                    result += part
                    z += 1

                  except IndexError:
                    pass

    try:
        return result / z

    except ZeroDivisionError:
        return result


def filter_err(obj, MammographMatrix):
  f = lambda x: filter(aug_signal(filter(aug_signal(x[-78:])))[::-1])[-80:]
  new_x = np.zeros((18, 18))
  matrix = MammographMatrix
  for i in range(18):
      for j in range(18):  
        if not matrix[i,j]:
          continue
        new_x[i, j] = np.sum(np.square(np.abs(obj[i,j] - np.mean(obj[i,j])) - f(obj[i,j])))
  return new_x


def sinus_mean(obj, MammographMatrix):
  new_x = np.zeros((18, 18))
  matrix = MammographMatrix
  for i in range(18):
      for j in range(18):  
        if not matrix[i,j]:
          continue
        new_x[i, j] = np.mean(obj[i,j])
  return new_x