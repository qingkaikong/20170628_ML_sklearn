############################### Solution 1 #################################

# fit a PCA model
pca = PCA().fit(X)

plt.figure(figsize=(10,8))
plt.plot(range(1, X.shape[1]+1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# we can see that with the first two components, we can explain over 95% of the variance, 
# then we can train PCA with only 2 components
# fit a PCA model
pca = PCA(n_components = 2).fit(X)
X_reduced = pca.transform(X)

############################### Solution 2 #################################
## Or we can simplely use n_components = 0.95
pca = PCA(n_components = 0.95).fit(X)
X_reduced = pca.transform(X)