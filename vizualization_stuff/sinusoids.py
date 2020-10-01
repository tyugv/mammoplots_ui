import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import fftpack


color_mx = np.array([['#00000000', '#00000000', '#00000000', '#00ff5503', '#00000000',
        '#00000000', '#42ff002a', '#5cff0063', '#76ff0085', '#96ff0086',
        '#b4ff0066', '#cdff002e', '#00000000', '#00000000', '#ffaa0003',
        '#00000000', '#00000000', '#00000000'],
       ['#00000000', '#00000000', '#00ff5503', '#00000000', '#04ff0038',
        '#1bff00b3', '#37ff00f6', '#5aff00ff', '#7bff00ff', '#9dff00ff',
        '#bcff00ff', '#d5ff00f8', '#f5ff00b9', '#fff70040', '#00000000',
        '#ffaa0003', '#00000000', '#00000000'],
       ['#00000000', '#00ff5503', '#00000000', '#00ff2372', '#01ff10fe',
        '#0fff00ff', '#2cff00ff', '#4dff00fd', '#71ff00fc', '#96ff00fc',
        '#baff00fd', '#deff00ff', '#ffff00ff', '#fff200ff', '#fcce007f',
        '#00000000', '#ffaa0003', '#00000000'],
       ['#00ff5503', '#00000000', '#00ff4b6f', '#00ff3eff', '#00ff20fd',
        '#03ff08fc', '#1cff00ff', '#43ff00ff', '#6eff00ff', '#9aff00ff',
        '#c4ff00ff', '#ebfe00ff', '#fdef00fc', '#ffd500fc', '#ffcc00ff',
        '#ffa5007e', '#00000000', '#ffaa0003'],
       ['#00ff0001', '#00ff712f', '#00ff66fd', '#00ff52fc', '#00ff39fd',
        '#00ff1cff', '#0aff02ff', '#35ff00ff', '#68ff00ff', '#9eff00ff',
        '#d2ff00ff', '#f9fa00ff', '#ffdb00ff', '#ffba00fe', '#ffa400fb',
        '#ff9800ff', '#ff7f003c', '#00000000'],
       ['#00000000', '#00ff8aaf', '#00ff89ff', '#00ff6dfb', '#00ff58ff',
        '#00ff3bff', '#00ff13ff', '#1fff00ff', '#60ff00ff', '#a4ff00ff',
        '#e5ff00ff', '#ffe600ff', '#ffbc00ff', '#ff9e00ff', '#ff8600fc',
        '#ff7b00fe', '#ff6800be', '#00000000'],
       ['#00ffb21e', '#00ffa7f5', '#00ff9fff', '#00ff90fe', '#00ff7dff',
        '#00ff63ff', '#00ff3cff', '#07ff09ff', '#4efe02ff', '#b2fd01ff',
        '#f9f400ff', '#ffbe00ff', '#ff9400ff', '#ff7900ff', '#ff6500ff',
        '#ff5700fe', '#ff4b00fb', '#ff3f002c'],
       ['#00ffc757', '#00ffceff', '#00ffbefd', '#00ffb6ff', '#00fea9ff',
        '#00fe96ff', '#01fe79ff', '#00fd3bff', '#2cff00ff', '#caff00ff',
        '#ffc601ff', '#fd8100ff', '#ff6100ff', '#ff4c00ff', '#ff3e00ff',
        '#ff3400fd', '#ff2f00ff', '#ff290068'],
       ['#00ffe776', '#00fff2ff', '#00ffe3fc', '#00ffe2ff', '#00ffdeff',
        '#00ffd8ff', '#01ffceff', '#00ffbaff', '#0dfe64ff', '#e1af01ff',
        '#ff4700ff', '#fd2b00ff', '#ff1e00ff', '#ff1700ff', '#ff1201ff',
        '#ff0f01fc', '#ff0d01ff', '#ff0b0188'],
       ['#00f8fc75', '#00feffff', '#00f4fdfc', '#00f2ffff', '#00edffff',
        '#00e5ffff', '#01d7ffff', '#00bcffff', '#1251ffff', '#e400b7ff',
        '#ff005aff', '#fd0040ff', '#ff0034ff', '#ff002dff', '#ff0029ff',
        '#ff0025fc', '#ff0025ff', '#ff002287'],
       ['#00ddff54', '#00e3ffff', '#00d1fffd', '#00c8ffff', '#00baffff',
        '#00a5ffff', '#0185feff', '#0043feff', '#2600fdff', '#bf00ffff',
        '#ff02d7ff', '#fe0197ff', '#ff0177ff', '#ff0163ff', '#ff0056ff',
        '#ff004bfd', '#ff0047ff', '#ff004165'],
       ['#00c6ff1b', '#00bbfff2', '#00b3fffe', '#00a1fffe', '#008effff',
        '#0073ffff', '#004bffff', '#0211ffff', '#3f01ffff', '#a001fdff',
        '#ef00fcff', '#ff00d3ff', '#fe00a9ff', '#ff008fff', '#ff007bff',
        '#ff006efe', '#ff0063f9', '#ff005928'],
       ['#00000000', '#009dffa8', '#009effff', '#0080fffb', '#006affff',
        '#004cffff', '#0022ffff', '#1001ffff', '#4e00ffff', '#9100ffff',
        '#d200ffff', '#fb00f6ff', '#ff00d2ff', '#ff00b4ff', '#ff009dfb',
        '#ff0093ff', '#ff007eb7', '#00000000'],
       ['#00ffff01', '#0085ff28', '#007afffa', '#0066fffc', '#004cfffd',
        '#002effff', '#020bffff', '#2200ffff', '#5500ffff', '#8a00ffff',
        '#bc00ffff', '#ea00ffff', '#ff00efff', '#ff00d1fd', '#ff00bbfb',
        '#ff00adff', '#ff009834', '#00000000'],
       ['#0055ff03', '#00000000', '#005dff62', '#0052ffff', '#0034ffff',
        '#0017fffb', '#0b02fffe', '#3000ffff', '#5a00ffff', '#8500ffff',
        '#af00ffff', '#d600fffe', '#f500fcfb', '#ff00ecfd', '#ff00e2ff',
        '#ff00bd71', '#00000000', '#ff00aa03'],
       ['#00000000', '#0055ff03', '#00000000', '#0038ff64', '#0023fff7',
        '#0309ffff', '#1800ffff', '#3900fffe', '#5d00fffc', '#8200fffc',
        '#a500fffd', '#c900ffff', '#f500ffff', '#fc00fdfc', '#ff00e570',
        '#00000000', '#ff00ff03', '#00000000'],
       ['#00000000', '#00000000', '#0055ff03', '#00000000', '#0011ff2c',
        '#0903ffa5', '#2200ffed', '#4300ffff', '#6400ffff', '#8600ffff',
        '#a400ffff', '#c000fff0', '#e100ffab', '#f000ff33', '#00000000',
        '#ff00aa03', '#00000000', '#00000000'],
       ['#00000000', '#00000000', '#00000000', '#0055ff03', '#00000000',
        '#00000000', '#2b00ff1d', '#4700ff52', '#6100ff73', '#7f00ff74',
        '#9f00ff55', '#b900ff21', '#00000000', '#00000000', '#ff00ff03',
        '#00000000', '#00000000', '#00000000']], dtype='<U32')

def sinusoid_plot(obj, MammographMatrix, x, y, act, color_mx):

  matrix = MammographMatrix
  fig = plt.figure(figsize=(20,5))

  q = obj.shape[4] // 10
  print(q)

  for i in range(18):
    for j in range(18):

        if matrix[i, j] == -1:
            continue

        if act == 'l':
          sins = obj[x, y, i, j]
          sins = sins-(sum(sins)/len(sins))


        else: 
          sins = obj[i, j, x, y]
          sins = sins-(sum(sins)/len(sins))
        
        #idx = np.argmax(sins[:q])
        #sins = sins[idx:-q]
        plt.plot(sins, color = color_mx[i][j], alpha = 0.2)

  return plt

def sinusoid_plot_norm(obj, MammographMatrix, x, y, act, color_mx):

  sorted_meas = np.sort(obj, axis=4)

  q = obj.shape[4] // 10

  amplitude = np.mean(sorted_meas[:, :, :, :, -q:], axis=4) - np.mean(sorted_meas[:, :, :, :, :q], axis=4)

  matrix = MammographMatrix
  fig = plt.figure(figsize=(20,5))

  for i in range(18):
    for j in range(18):

        if matrix[i, j] == -1:
            continue

        if act == 'l':
          sins = obj[x, y, i, j]
          sins = sins-(sum(sins)/len(sins))
          sins = sins/amplitude[x, y, i, j]

        else: 
          sins = obj[i, j, x, y]
          sins = sins-(sum(sins)/len(sins))
          sins = sins/amplitude[i, j, x, y]
        
        #idx = np.argmax(sins[:q])
        #sins = sins[idx:-q]
        plt.plot(sins, color = color_mx[i][j], alpha = 0.2)

  return plt

def FourierPlot(sins, size = (5, 4)):

  fig = plt.figure(figsize=size)

  N = len(sins)

  fft = fftpack.fft(sins) 
  spectrum = 2/N * np.abs(fft[:int(N/2)]) # positive freqs only

  plt.title('Fourier Transform')
  plt.grid()
  plt.stem(spectrum, use_line_collection=True, basefmt='C0')

  return plt
