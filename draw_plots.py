#from vizualization_stuff.distplots import make_distplot_gif, dist_to_3d, dist_of_array
#from vizualization_stuff.graph_3d import make_my_plot
from vizualization_stuff.analytics_of_errors import sub_deviance, deviance, matrix_voltage_error, sub_distance_meas, meas
from vizualization_stuff.sinusoids import color_mx, sinusoid_plot_norm, sinusoid_plot, FourierPlot

from amplitude import meas_to_x
from mammo_packets import read_from_file_binary, parse_mammograph_raw_data, mammograph_matrix
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


BINSDIR = 'vizualizations/bins'

class MammographMatrix:
    def __init__(self):
        self.matrix = np.zeros((18, 18), dtype=np.int32) - 1
        self.matrix_inverse = np.zeros((18, 18), dtype=np.int32) + 1
        gen = iter(range(256))

        for i in range(6, 18 - 6):
            self.matrix[0, i] = next(gen)

        for i in range(4, 18 - 4):
            self.matrix[1, i] = next(gen)

        for i in range(3, 18 - 3):
            self.matrix[2, i] = next(gen)

        for i in range(2, 18 - 2):
            self.matrix[3, i] = next(gen)

        for j in range(2):
            for i in range(1, 18 - 1):
                self.matrix[4 + j, i] = next(gen)

        for j in range(6):
            for i in range(18):
                self.matrix[6 + j, i] = next(gen)

        for j in range(2):
            for i in range(1, 18 - 1):
                self.matrix[12 + j, i] = next(gen)

        for i in range(2, 18 - 2):
            self.matrix[14, i] = next(gen)

        for i in range(3, 18 - 3):
            self.matrix[15, i] = next(gen)

        for i in range(4, 18 - 4):
            self.matrix[16, i] = next(gen)

        for i in range(6, 18 - 6):
            self.matrix[17, i] = next(gen)

        for i in range(18):
            for j in range(18):
                if self.matrix[i, j] != -1:
                    self.matrix_inverse[i, j] = 0

mammograph_matrix = MammographMatrix().matrix
parser = parse_mammograph_raw_data

def txt_file_to_x(path, mammograph_matrix):
	with open(path, encoding='cp1251') as f:
		need_check = True
		lst = []
		for line in f.readlines():
			if need_check and line.count('0;') != 0:
				need_check = False
			elif not need_check:
				pass
			else:
				continue

			one_x = np.zeros((18, 18))
			line = line[:-2].split(';')

			for i in range(18):
				for j in range(18):
					one_x[i, j] = int(line[i * 18 + j])
			lst.append(one_x)

		x = np.zeros((18, 18, 18, 18))

		for i in range(18):
			for j in range(18):
				if mammograph_matrix[i, j] != -1:
					x[i, j] = lst[mammograph_matrix[i, j] - 1]
	return x

def save_plot(plot, path):
	plot.savefig(path)
	plot.close()

def save_plot_img(img, path, title = '', cmap = 'hot'):
	fig = plt.figure(figsize=(4,4))
	plt.imshow(img, cmap=cmap, interpolation='none')
	plt.title(title)
	save_plot(plt, path)

def draw_plots(fname):

	parser = parse_mammograph_raw_data

	if (fname[-4:]) == '.bin':
		data = read_from_file_binary(BINSDIR + f'/{fname}')
		arr = parser(data)
		x = meas_to_x(arr)
		x = x[0][0]

	elif (fname[-4:]) == '.txt':
		x = txt_file_to_x(BINSDIR + f'/{fname}', mammograph_matrix)

	#make_my_plot(x, figursize = (20,20), toSave = True, 
   # 	filename = f'vizualizations/images/{fname}/3dplot', euclid_colors = True)

	save_plot_img(matrix_voltage_error(x, mammograph_matrix),
		f'vizualizations/images/{fname}/matrix_voltage_error.png')

	for act in ['l', 'g']:
		for i in range(18):
			for j in range(18):
				if (act == 'l'):
					img1 = x[i, j, :, :]

				else:
					img1 = x[:, :, i, j]

				# возможно стоит поменять аргументы
				img2 = deviance(x, mammograph_matrix, (i, j, act), rank = 1)
				img3 = meas(x, mammograph_matrix, (i, j, act), rank =1)

				save_plot_img(img1, 
					f'vizualizations/images/{fname}/slice{i}_{j}_{act}.png', title = 'slice')

				save_plot_img(img2, 
					f'vizualizations/images/{fname}/deviance{i}_{j}_{act}.png', title = 'deviance')

				save_plot_img(img3, 
					f'vizualizations/images/{fname}/meas{i}_{j}_{act}.png', title = 'meas')


	#IMGSNAMES = os.listdir(f'vizualizations/images/{fname}')
	#with ZipFile(f'vizualizations/images/{fname}/{fname[:-4]}.zip', 'w') as zipObj:
	#	for imgname in IMGSNAMES:
	#		zipObj.write(f'vizualizations/images/{fname}/{imgname}', fname + '/' + imgname)

def get_matrix(path):

	if (path[-4:]) == '.bin':

		data = read_from_file_binary(path)
		x = parser(data)

	elif (path[-4:]) == '.txt':

		x = txt_file_to_x(path, mammograph_matrix)

	return x

def draw_big_plots(meas_m, folder):

	if (meas_m.shape[-1] == 80):
		x = meas_to_x(meas_m)
		x = x[0][0]

		save_plot_img([[x[i,j,i,j] for j in range(18)] for i in range(18)], path = f'{folder}/selfhot.png', title = 'slice (i,j,i,j) in hot')
		save_plot_img([[x[i,j,i,j] for j in range(18)] for i in range(18)], path = f'{folder}/selfviridis.png', title = 'slice (i,j,i,j) in viridis', cmap = 'viridis')

	else:

		x = meas_m
		save_plot_img([[x[i,j,i,j] for j in range(18)] for i in range(18)], path = f'{folder}/selfhot.png', title = 'slice (i,j,i,j) in hot')
		save_plot_img([[x[i,j,i,j] for j in range(18)] for i in range(18)], path = f'{folder}/selfviridis.png', title = 'slice (i,j,i,j) in viridis', cmap = 'viridis')

	save_plot_img(matrix_voltage_error(x, mammograph_matrix), path = f'{folder}/matrix_voltage_error.png', title = 'matrix voltage error')


def draw_elements_plots(meas_m, folder, i, j, act):

	if (meas_m.shape[-1] == 80):
		x = meas_to_x(meas_m)
		x = x[0][0]

		#save_plot(sinusoid_plot(meas_m, mammograph_matrix, i, j, act, color_mx), 
		#f'{folder}/sinusoid{i}_{j}_{act}.png')

		#save_plot(sinusoid_plot_norm(meas_m, mammograph_matrix, i, j, act, color_mx), 
		#f'{folder}/sinusoid_ampl{i}_{j}_{act}.png')

	else:
		x = meas_m


	if (act == 'l'):

		img1 = x[i, j, :, :]

	else:
		img1 = x[:, :, i, j]

	img2 = deviance(x, mammograph_matrix, (i, j, act), rank = 1)
	img3 = meas(x, mammograph_matrix, (i, j, act), rank =1)

	img1[i,j] = 0
	img2[i,j] = 0
	img3[i,j] = 0

	save_plot_img(img1, 
		f'{folder}/slice{i}_{j}_{act}.png', title = 'slice')

	save_plot_img(img2, 
		f'{folder}/deviance{i}_{j}_{act}.png', title = 'deviance')

	save_plot_img(img3, 
		f'{folder}/meas{i}_{j}_{act}.png', title = 'meas')



def draw_sinusoid(obj, x, y, i, j, folder):

	if (obj.shape[-1] != 80):

		return

	fig = plt.figure(figsize=(6,4))
	sins = obj[x,y,i,j]
	plt.plot(sins)
	plt.title('Sinusoid plot')
	save_plot(plt, f'{folder}/one_sinusoid{x}_{y}_{i}_{j}.png')

	save_plot(FourierPlot(obj[x,y,i,j], size = (5, 3)), f'{folder}/sinusoid_fft{x}_{y}_{i}_{j}.png')
