"""
	Takes log files of multiple models and plots the train and validation losses
	X axis denotes number of interations
	
	Parameters:
		windowVal - moving average window size for validation
		windowTrain - moving average size for train

	Usage:
		python plot_loss.py [model_log_file]*

	Note:
		'plotname' will be the name of the first model
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os

"""
	Returns moving average of 'loss' array, where 'window' is the size of the moving window
	Assuming that 'loss' will have at least 'window' elements
"""
def movingAverage(loss, window):
	mas = []
	for i in range(len(loss)):
		j = i - window + 1
		if (j < 0):
			j = 0
		sum = 0.0
		for k in range(j, i + 1):
			sum += loss[k]
		mas.append(sum / (i - j + 1.0))
	return mas

plotname = sys.argv[1]
while (plotname[:3] == '../'):
	plotname = plotname[-(len(plotname) - 3):]
plotname = plotname + '.png'

windowVal = 50
windowTrain = 1500

minv = 1e8
maxv = -1e8

""" 
	Plots train and validation losses for a single model
	'filename' is filename of model's log file
	'index' is used to choose the plot color
"""
def plotTrainVal(filename, index):
	global minv
	global maxv
	
	os.system("egrep 'Iteration.*loss|Train net output' " + filename +  " | egrep 'Iteration [0-9]*| Train net output #0: loss = [0-9|.]*' -o >tmpLossTrain.txt")
	os.system("egrep 'Iteration.*Testing|Test net output' " + filename + " | egrep 'Iteration [0-9]*| Test net output #1: loss = [0-9|.]*' -o >tmpLossVal.txt")
	tmpLossVal = open('tmpLossVal.txt', 'r')
	tmpLossTrain = open('tmpLossTrain.txt', 'r')

	valx = []
	valy = []
	for st in tmpLossVal.readlines():
		if (st.split(' ')[0] == 'Iteration'):
			valx.append(int(st.split(' ')[1]))
		else:
			valy.append(float(st.split(' ')[7]))
	
	trainx = []
	trainy = []
	for st in tmpLossTrain.readlines():
		if (st.split(' ')[0] == 'Iteration'):
			trainx.append(int(st.split(' ')[1]))
		else:
			trainy.append(float(st.split(' ')[7]))

	os.remove('tmpLossVal.txt')
	os.remove('tmpLossTrain.txt')
	
	wndVal = min(windowVal, int(0.8 * len(valy)))
	wndTrain = min(windowTrain, int(0.8 * len(trainy)))
	
	print "Train length: ", len(trainy), " \t\t window: ", wndTrain
	print "Val length: ", len(valy), " \t\t window: ", wndVal
	
	valy = movingAverage(valy, wndVal)
	trainy = movingAverage(trainy, wndTrain)
	valx = valx[:len(valy)]
	trainx = trainx[:len(trainy)]
	
	
	plt.plot(trainx, trainy, '#0000' + hex(256 - index * 32)[2:])
	plt.hold(True)
	plt.plot(valx, valy, '#' + hex(256 - index * 32)[2:] + '0000')
	plt.hold(True)
	
	minv = min(minv, min(trainy))
	maxv = max(maxv, max(trainy))
	minv = min(minv, min(valy))
	maxv = max(maxv, max(valy))
	

for i in range(1, len(sys.argv)):
	plotTrainVal(sys.argv[i], i)

minv = minv * 0.8
maxv = maxv * 1.2
#plt.gca().set_yticks(np.linspace(minv, maxv, int((maxv - minv) * 20)), minor=True)
plt.gcf().savefig(plotname)
