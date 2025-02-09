import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


# building the correlogram based on the correlation matrix
def correlogram(R2, dec=2, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
               cmap='bwr', annot=True)
               # cmap='BuPu', annot=True)

#Principal components graph
def plot_variance(alpha, title='Eigenvalues - '
                       'explained variance by the principal components',
            labelX='Principal components',
                  labelY='Eigenvalues (explained variance)'):
    plt.figure(title,
               figsize=(11, 8))
    plt.title(title,
              fontsize=16, color='k', verticalalignment='bottom')
    plt.xlabel(labelX,
               fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(labelY,
               fontsize=14, color='k', verticalalignment='bottom')
    Xindex = ['C' + str(k + 1) for k in range(len(alpha))]
    plt.plot(Xindex, alpha, 'bo-')
    plt.xticks(Xindex)
    plt.axhline(1, color='r')

# building the correlation circle
# including concentric circles for the quartiles
def corr_circle_quartiles(R, k1, k2, radius=1, con=False,
                          xLabel=None, yLabel=None,
                 title='Correlation circle with quartiles',
                          valMin=-1, valMax=1):
    plt.figure(title, figsize=(10, 10))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    if xLabel == None or yLabel == None:
        if isinstance(R, pd.DataFrame):
            plt.xlabel(xlabel=R.columns[k1], fontsize=14,
                       color='k', verticalalignment='top')
            plt.ylabel(ylabel=R.columns[k2], fontsize=14,
                       color='k', verticalalignment='bottom')
        else:       # isinstance(R, np.ndarray):
            plt.xlabel(xlabel='Component ' + str(k1 + 1), fontsize=14,
                       color='k', verticalalignment='top')
            plt.ylabel(ylabel='Component ' + str(k2 + 1), fontsize=14,
                       color='k', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=xLabel, fontsize=14, color='k',
                   verticalalignment='top')
        plt.ylabel(ylabel=yLabel, fontsize=14, color='k',
                   verticalalignment='bottom')

    # building the circle of correlations, of radius 1,
    # with the center at the origin of the coordinate axes
    # generate a list of values for an angle that
    # takes values around the circle
    T = [t for t in np.arange(0, np.pi * 2, 0.01)]
    X = [np.cos(t) * radius for t in T]  # f(t) = cos(t)
    Y = [np.sin(t) * radius for t in T]  # f(t) = sin(t)
    plt.plot(X, Y)

    if con:
        for k in range(1, 3 + 1):
            X = [np.cos(t) * 0.25 * k * radius for t in T]
            Y = [np.sin(t) * 0.25 * k * radius for t in T]
            plt.plot(X, Y)

    plt.axhline(0, color='g')
    plt.axvline(0, color='g')
    if isinstance(R, pd.DataFrame):
        plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='r',
                    vmin=valMin, vmax=valMax)
        for i in range(len(R)):
            plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])
    else:   # isinstance(R, np.ndarray):
        plt.scatter(R[:, k1], R[:, k2], c='r', vmin=valMin, vmax=valMax)
        for i in range(len(R)):
            plt.text(R[i, k1], R[i, k2], '(' +
                     str(np.round(R[i, k1], 1)) +
                     ', ' +
                     str(np.round(R[i, k2], 1)) +
                     ')')

# creates the graphical table based on the intensity of link between
# values on X and Y axes
def intensity_map(R2, dec=1, title='Intensity Map', valmin=None, valmax=None,):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
               # cmap='Purples', annot=True)
               #  cmap = 'Reds', annot = True)
                cmap = 'Oranges', annot = True)
                # cmap = 'Blues', annot = True)

def show():
    plt.show()