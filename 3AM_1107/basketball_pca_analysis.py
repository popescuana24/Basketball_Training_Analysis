import graphicsPCA as graphics
import pandas as pd
import pca.PCA as pca

table = pd.read_csv('./dataIN/basketball_training_dataset_modified.csv', index_col=0)

observations = table.index[:]
variables = table.columns[:]
basketball_table = table[variables].values
print(basketball_table)

n = basketball_table.shape[0]
m = basketball_table.shape[1]
print(n,m)

#instantiate PCA object
pcaModel = pca.PCA(basketball_table)

# Get all PCA components
R = pcaModel.getCorr()
alpha = pcaModel.getEigenvalues()
a = pcaModel.getEigenvectors()
factorLoadings = pcaModel.getFactorLoadings()
C = pcaModel.getComponents()
scores = pcaModel.getScores()
obsQuality = pcaModel.getObsQuality()
beta = pcaModel.getBeta()
communalities = pcaModel.getCommun()
print(communalities)

# Save results to CSV files
C_df = pd.DataFrame(data=C,
                    columns=variables, index=observations)
C_df.to_csv(path_or_buf='./dataOUTPUT/PrincipalComponents.csv')

# save the standardised matrix X in a CSV file
Xstd_df = pd.DataFrame(data=pcaModel.getXstd(),
                    columns=variables, index=observations)
Xstd_df.to_csv(path_or_buf="dataOUTPUT/Xstd.csv")

# save the correlation matrix R
R_df = pd.DataFrame(data=R,
                     columns=variables, index=variables)
R_df.to_csv(path_or_buf='dataOUTPUT/R.csv')

# save in a CSV file the matrix of correlation factors
factorLoadings_df = pd.DataFrame(factorLoadings, index=variables,
                       columns=('C'+str(k+1) for k in range(m)))
factorLoadings_df.to_csv(path_or_buf='./dataOUTPUT/FactorLoadings.csv')

# create the correlogram of factor loadings
graphics.correlogram(factorLoadings_df, dec=2, valmin=-1, valmax=1,
                     title="Correlogram of factor loadings")

# create the correlogram of correlation matrix
graphics.correlogram(R_df, valmin=-1, valmax=1,
                     title='Correlation matrix of causal variables')

# create the graphic of eigenvalues
graphics.plot_variance(alpha)

# save the matrix of communalities in a CSV file
communalities_df = pd.DataFrame(communalities, index=variables,
                                columns=('C'+str(k+1) for k in range(m)))
communalities_df.to_csv(path_or_buf='./dataOUTPUT/Communalities.csv')

# graph of communalities
graphics.intensity_map(communalities_df, dec=1,
                       title='Communalities of the principal components found in the causal variables')

# create the correlation circle of the causal
# variables in the space of components 1 and 2
graphics.corr_circle_quartiles(factorLoadings_df, 0, 1, con=True,
            title='Observed variables in the space of principal components 1 and 2')

graphics.show()