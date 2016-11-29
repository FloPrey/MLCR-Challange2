'''
Created on Nov 29, 2016

@author: tobi
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import Challenge2.DatasetCreator as dc
import pylab

    
def do3DPlot(inputData, outputLabel):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x =inputData["time_as_float"].tolist()
    y =inputData["positive_mean"].tolist()
    z =outputLabel
    
    
    
    ax.scatter(x, y, z, c='r', marker='o')
    
    ax.set_xlabel('Time Of Test')
    ax.set_ylabel('Mean')
    ax.set_zlabel('MSF')
    
    plt.show()
    

def do2DPlot(inputData, outputLabel):

    x =inputData["time_as_float"].tolist()
    y =outputLabel

    colors = inputData["Participant_ID"].tolist()
    pylab.scatter(x, y, c=colors)
    pylab.show()

dataset = dc.createDataSet()

inputData, outputLabel = dc.createInputAndOutputDataset(dataset)

do3DPlot(inputData, outputLabel)
do2DPlot(inputData, outputLabel)