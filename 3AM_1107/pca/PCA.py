import numpy as np

class PCA:
    def __init__(self, X):
        self.X = X
        self.R = np.corrcoef(self.X, rowvar=False)

        '''
        standardize the initial matrix X,
        due to the weight of the causal variables,
        to not introduce interference in the linear combinations
        that make up the main components
        '''
        avgs = np.mean(self.X, axis=0)  # means on the columns
        stds = np.std(self.X, axis=0)  # standard deviation on the columns
        self.Xstd = (self.X - avgs) / stds

        # calculation of the variance-covariance matrix for Xstd,
        self.Cov = np.cov(self.Xstd, rowvar=False)

        # calculation of eigenvalues and eigenvectors
        # for the variance-covariance matrix Cov = 1/n * (X)t * X
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.Cov)

        # sort in descending order of the eigenvalues,
        # along with the eigenvectors
        k_des = [k for k in reversed(np.argsort(self.eigenvalues))]
        self.alpha = self.eigenvalues[k_des]
        self.a = self.eigenvectors[:, k_des]

        '''
        eigenvector regularization - eigenvectors values
        are relevant from the explained variance perspective;
        if there is a larger negative value in an eigenvector,
        as absolute value, than the positive ones, then it is
        more useful for interpreting the results to
        reverse the sign of the values of the eigenvector in case;
        the nature of the eigenvector is preserved by
        multiplying it with a scalar, respectively -1
        '''
        for j in range(len(self.alpha)):
            minim = np.min(self.a[:, j])
            maxim = np.max(self.a[:, j])
            if np.abs(minim) > np.abs(maxim):
                self.a[:, j] = -self.a[:, j]

        '''
        calculation of the principal components
        for the standardized matrix X;
        in the numpy package the @ operator is overloaded for
        matrix multiplication, equivalent to the following instruction:
        self.C = np.matmul(self.Xstd, self.a)
        '''
        self.C = self.Xstd @ self.a

        # compute the correlation factors
        self.Rxc = self.a * np.sqrt(self.alpha)

        # self.C2 = np.square(self.C)
        self.C2 = self.C * self.C

    # return the correlation matrix R,
    # corresponding to the initial matrix X
    def getCorr(self):
        return self.R

    # return the standardised initial matrix X
    def getXstd(self):
        return self.Xstd

    # return the variance-covariance matrix Cov,
    # corresponding to the initial standardised matrix Xstd
    def getCov(self):
        return self.Cov

    # return the eigenvalues of the correlation matrix R
    def getEigenvalues(self):
        return self.alpha

    # return th eigenvectors of teh correlation matrix R
    def getEigenvectors(self):
        return self.a

    # return the matrix of correlation factors (factor loadings),
    # the correlation between the observed variables and
    # the principal components
    def getFactorLoadings(self):
        return self.Rxc

    # return the principal components
    def getComponents(self):
        return self.C

    # compute and return the scores,
    # standardised principal components
    def getScores(self):
        return self.C / np.sqrt(self.alpha)

    # compute and return quality of points (observations)
    # representation on the axes of the principal components
    def getObsQuality(self):
        C2Sum = np.sum(self.C2, axis=1)
        return np.transpose(np.transpose(self.C2) / C2Sum)

    # compute and return the observations (instances)
    # contribution to the variance of axes
    def getBeta(self):
        return self.C2 / (self.alpha * self.X.shape[0])

    # return the communalities of the principal components
    # found in the observed (causal) variables
    def getCommun(self):
        R2 = np.square(self.Rxc)
        return np.cumsum(R2, axis=1)
