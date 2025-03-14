# import numpy as np
# from scipy.spatial.distance import pdist, squareform


# class GaussianProcess:
#     """
#     The crop yield Gaussian process
#     """
#     def __init__(self, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01):
#         self.sigma = sigma
#         self.r_loc = r_loc
#         self.r_year = r_year
#         self.sigma_e = sigma_e
#         self.sigma_b = sigma_b

#     @staticmethod
#     def _normalize(x):
#         x_mean = np.mean(x, axis=0, keepdims=True)
#         x_scale = np.ptp(x, axis=0, keepdims=True)

#         return (x - x_mean) / x_scale

#     def run(self, feat_train, feat_test, loc_train, loc_test, year_train, year_test,
#             train_yield, model_weights, model_bias):

#         # makes sure the features have an additional testue for the bias term
#         # We call the features H since the features are used as the basis functions h(x)
#         H_train = np.concatenate((feat_train, np.ones((feat_train.shape[0], 1))), axis=1)
#         H_test = np.concatenate((feat_test, np.ones((feat_test.shape[0], 1))), axis=1)

#         Y_train = np.expand_dims(train_yield, axis=1)

#         n_train = feat_train.shape[0]
#         n_test = feat_test.shape[0]

#         locations = self._normalize(np.concatenate((loc_train, loc_test), axis=0))
#         years = self._normalize(np.concatenate((year_train, year_test), axis=0))
#         # to calculate the se_kernel, a dim=2 array must be passed
#         years = np.expand_dims(years, axis=1)

#         # These are the squared exponential kernel function we'll use for the covariance
#         se_loc = squareform(pdist(locations, 'euclidean')) ** 2 / (self.r_loc ** 2)
#         se_year = squareform(pdist(years, 'euclidean')) ** 2 / (self.r_year ** 2)

#         # make the dirac matrix we'll add onto the kernel function
#         noise = np.zeros([n_train + n_test, n_train + n_test])
#         noise[0: n_train, 0: n_train] += (self.sigma_e ** 2) * np.identity(n_train)

#         kernel = ((self.sigma ** 2) * np.exp(-se_loc) * np.exp(-se_year)) + noise

#         # since B is diagonal, and B = self.sigma_b * np.identity(feat_train.shape[1]),
#         # its easy to calculate the inverse of B
#         B_inv = np.identity(H_train.shape[1]) / self.sigma_b
#         # "We choose b as the weight vector of the last layer of our deep models"
#         b = np.concatenate((model_weights.transpose(1, 0), np.expand_dims(model_bias, 1)))

#         K_inv = np.linalg.inv(kernel[0: n_train, 0: n_train])

#         # The definition of beta comes from equation 2.41 in Rasmussen (2006)
#         beta = np.linalg.inv(B_inv + H_train.T.dot(K_inv).dot(H_train)).dot(
#             H_train.T.dot(K_inv).dot(Y_train) + B_inv.dot(b))

#         # We take the mean of g(X*) as our prediction, also from equation 2.41
#         pred = H_test.dot(beta) + \
#                kernel[n_train:, :n_train].dot(K_inv).dot(Y_train - H_train.dot(beta))

#         return pred


import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class GaussianProcess:
    """
    The crop yield Gaussian process
    """
    def __init__(self, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
                 use_sparse=False, num_inducing=500, sparse_method='fitc'):
        self.sigma = sigma
        self.r_loc = r_loc
        self.r_year = r_year
        self.sigma_e = sigma_e
        self.sigma_b = sigma_b
        
        # Sparse GP parameters
        self.use_sparse = use_sparse
        self.num_inducing = num_inducing
        self.sparse_method = sparse_method
        
        if self.sparse_method not in ['fitc', 'vfe']:
            raise ValueError("Sparse method must be either 'fitc' or 'vfe'")

    @staticmethod
    def _normalize(x):
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_scale = np.ptp(x, axis=0, keepdims=True)
        return (x - x_mean) / (x_scale + 1e-8)  # Add small constant to avoid division by zero

    def _select_inducing_points(self, locations, years):
        """
        Select inducing points using k-means clustering on combined location and year features
        """
        print("Selecting inducing points...")
        # Scale years to have similar impact as locations
        year_scale = (self.r_loc / self.r_year)
        years_scaled = years * year_scale
        
        # Combine features for clustering
        combined = np.concatenate([locations, years_scaled], axis=1)
        
        # Limit the number of inducing points to the number of samples
        m = min(self.num_inducing, combined.shape[0])
        
        # Use k-means clustering to find representative inducing points
        kmeans = KMeans(n_clusters=m, random_state=42, n_init=10)
        kmeans.fit(combined)
        
        # Extract inducing points
        Z = kmeans.cluster_centers_
        Z_loc = Z[:, :-1]  # All but the last column
        Z_year = Z[:, -1:] / year_scale  # Last column, undo scaling
        
        return Z_loc, Z_year
    
    def _compute_kernel(self, X_loc1, X_year1, X_loc2=None, X_year2=None):
        """
        Compute squared exponential kernel between two sets of points
        """
        if X_loc2 is None and X_year2 is None:
            # Compute kernel for single set of points
            se_loc = squareform(pdist(X_loc1, 'euclidean')) ** 2 / (self.r_loc ** 2)
            se_year = squareform(pdist(X_year1, 'euclidean')) ** 2 / (self.r_year ** 2)
        else:
            # Compute kernel between two sets of points
            se_loc = cdist(X_loc1, X_loc2, 'euclidean') ** 2 / (self.r_loc ** 2)
            se_year = cdist(X_year1, X_year2, 'euclidean') ** 2 / (self.r_year ** 2)
        
        # Return the full kernel
        return (self.sigma ** 2) * np.exp(-se_loc) * np.exp(-se_year)
    
    # def _compute_diag_kernel(self, X_loc):
    #     """Compute diagonal of kernel matrix (more efficient than full matrix)"""
    #     # For squared exponential kernel with same points, diagonal is constant
    #     return np.ones(X_loc.shape[0]) * (self.sigma ** 2)
    
    def _compute_diag_kernel(self, X_loc):
        """Compute diagonal of kernel matrix (more efficient than full matrix)"""
        # For squared exponential kernel with same points, diagonal is constant
        return np.ones(X_loc.shape[0], dtype=np.float32) * (self.sigma ** 2)

    def run_original(self, feat_train, feat_test, loc_train, loc_test, year_train, year_test,
                    train_yield, model_weights, model_bias):
        """Original full GP implementation"""
        
        # makes sure the features have an additional feature for the bias term
        # We call the features H since the features are used as the basis functions h(x)
        H_train = np.concatenate((feat_train, np.ones((feat_train.shape[0], 1))), axis=1)
        H_test = np.concatenate((feat_test, np.ones((feat_test.shape[0], 1))), axis=1)

        Y_train = np.expand_dims(train_yield, axis=1)

        n_train = feat_train.shape[0]
        n_test = feat_test.shape[0]

        locations = self._normalize(np.concatenate((loc_train, loc_test), axis=0))
        years = self._normalize(np.concatenate((year_train, year_test), axis=0))
        # to calculate the se_kernel, a dim=2 array must be passed
        years = np.expand_dims(years, axis=1)

        # These are the squared exponential kernel function we'll use for the covariance
        se_loc = squareform(pdist(locations, 'euclidean')) ** 2 / (self.r_loc ** 2)
        se_year = squareform(pdist(years, 'euclidean')) ** 2 / (self.r_year ** 2)

        # make the dirac matrix we'll add onto the kernel function
        noise = np.zeros([n_train + n_test, n_train + n_test])
        noise[0: n_train, 0: n_train] += (self.sigma_e ** 2) * np.identity(n_train)

        kernel = ((self.sigma ** 2) * np.exp(-se_loc) * np.exp(-se_year)) + noise

        # since B is diagonal, and B = self.sigma_b * np.identity(feat_train.shape[1]),
        # its easy to calculate the inverse of B
        B_inv = np.identity(H_train.shape[1]) / self.sigma_b
        # "We choose b as the weight vector of the last layer of our deep models"
        b = np.concatenate((model_weights.transpose(1, 0), np.expand_dims(model_bias, 1)))

        K_inv = np.linalg.inv(kernel[0: n_train, 0: n_train])

        # The definition of beta comes from equation 2.41 in Rasmussen (2006)
        beta = np.linalg.inv(B_inv + H_train.T.dot(K_inv).dot(H_train)).dot(
            H_train.T.dot(K_inv).dot(Y_train) + B_inv.dot(b))

        # We take the mean of g(X*) as our prediction, also from equation 2.41
        pred = H_test.dot(beta) + \
               kernel[n_train:, :n_train].dot(K_inv).dot(Y_train - H_train.dot(beta))

        return pred

    def run_sparse(self, feat_train, feat_test, loc_train, loc_test, year_train, year_test,
               train_yield, model_weights, model_bias, debug_viz=False):
        """Sparse GP implementation"""
        
        max_features = 100  # Try 100 instead of 1000/2000
    
        feat_train = feat_train.astype(np.float32)
        feat_test = feat_test.astype(np.float32)
        loc_train = loc_train.astype(np.float32)
        loc_test = loc_test.astype(np.float32)
        year_train = year_train.astype(np.float32)
        year_test = year_test.astype(np.float32)
        train_yield = train_yield.astype(np.float32)
        
        if feat_train.shape[1] > max_features:
            from sklearn.random_projection import GaussianRandomProjection
            print(f"Reducing feature dimension from {feat_train.shape[1]} to {max_features}")
            transformer = GaussianRandomProjection(n_components=max_features, random_state=42)
            feat_train = transformer.fit_transform(feat_train)
            feat_test = transformer.transform(feat_test)
        
        # Add a feature for the bias term
        H_train = np.concatenate((feat_train, np.ones((feat_train.shape[0], 1))), axis=1)
        H_test = np.concatenate((feat_test, np.ones((feat_test.shape[0], 1))), axis=1)
        
        Y_train = np.expand_dims(train_yield, axis=1)
        
        n_train = feat_train.shape[0]
        n_test = feat_test.shape[0]
        
        # Normalize the locations and years
        locations_all = self._normalize(np.concatenate((loc_train, loc_test), axis=0))
        years_all = self._normalize(np.concatenate((year_train, year_test), axis=0))
        years_all = np.expand_dims(years_all, axis=1)  # Make it 2D
        
        # Split back into train/test
        locations_train = locations_all[:n_train]
        locations_test = locations_all[n_train:]
        years_train = years_all[:n_train]
        years_test = years_all[n_train:]
        
        # Select inducing points
        Z_loc, Z_year = self._select_inducing_points(locations_all, years_all)
        m = Z_loc.shape[0]  # Number of inducing points
        
        if debug_viz and locations_train.shape[1] == 2:
            # Simple visualization of inducing points and data points
            plt.figure(figsize=(10, 8))
            plt.scatter(locations_train[:, 0], locations_train[:, 1], c='blue', s=20, alpha=0.5, label='Training points')
            plt.scatter(locations_test[:, 0], locations_test[:, 1], c='green', s=20, alpha=0.5, label='Test points')
            plt.scatter(Z_loc[:, 0], Z_loc[:, 1], c='red', s=50, marker='x', label='Inducing points')
            plt.legend()
            plt.title('Data points and inducing points')
            plt.savefig('/Users/akshat/Developer/ML_WORK/pycrop-yield-prediction/data/data_and_inducing.png')
        
        print("Computing kernel matrices...")
        # Compute kernel matrices
        K_mm = self._compute_kernel(Z_loc, Z_year)
        K_mm += 1e-8 * np.eye(m)  # Add jitter for numerical stability
        
        K_nm = self._compute_kernel(locations_train, years_train, Z_loc, Z_year)
        K_mn = K_nm.T
        
        K_star_m = self._compute_kernel(locations_test, years_test, Z_loc, Z_year)
        
        # Construct b vector (weights from deep model)
        print(f"H_train shape: {H_train.shape}, model_weights shape: {model_weights.shape}")

        # Ensure b has compatible shape with H_train for matrix multiplication
        feature_dim = H_train.shape[1] - 1  # Number of features (excluding bias)

        # Reshape weights to be compatible
        if len(model_weights.shape) > 1:
            weights = model_weights.flatten()[:feature_dim]
        else:
            weights = model_weights[:feature_dim]
            
        # If weights aren't sufficient length, pad with zeros
        if weights.size < feature_dim:
            weights = np.pad(weights, (0, feature_dim - weights.size))
            
        # Ensure model_bias is 2-dimensional
        model_bias = np.array(model_bias).reshape(-1, 1)
        
        # Create the b vector with proper shape
        b = np.vstack([weights.reshape(-1, 1), model_bias])
        
        print("Computing sparse approximation...")
        if self.sparse_method == 'fitc':
            # FITC approximation (Snelson & Ghahramani, 2006)
            K_nn_diag = self._compute_diag_kernel(locations_train)
            diag_correction = K_nn_diag - np.sum(K_nm * np.linalg.solve(K_mm, K_nm.T).T, axis=1)
            
            # Add noise and diagonal correction
            Lambda = np.diag(diag_correction + self.sigma_e**2)
            
            # Compute posterior using Woodbury identity for efficiency
            sigma_b_inv = 1.0 / self.sigma_b
            H_train_T = H_train.T
            
            # Compute intermediate terms for numerical stability
            L = np.linalg.cholesky(K_mm + K_mn @ np.linalg.solve(Lambda, K_nm))
            c = np.linalg.solve(Lambda, Y_train - H_train @ b)
            v = np.linalg.solve(L, K_mn @ c)
            
            # Update b with posterior information
            B_inv = np.eye(H_train.shape[1]) * sigma_b_inv
            beta = np.linalg.solve(
                B_inv + H_train_T @ np.linalg.solve(Lambda, H_train),
                H_train_T @ np.linalg.solve(Lambda, Y_train) + B_inv @ b
            )
            
            # Compute mean prediction
            pred = H_test @ beta + K_star_m @ np.linalg.solve(K_mm, v)
            
        else:  # 'vfe' method
            # VFE approximation (Titsias, 2009)
            # Cholesky decomposition for numerical stability
            L_mm = np.linalg.cholesky(K_mm)
            
            # Compute intermediate matrices
            Lm_inv_Kmn = np.linalg.solve(L_mm, K_mn)
            
            # Prior precision and regularization
            sigma_inv = 1.0 / self.sigma_e**2
            sigma_b_inv = 1.0 / self.sigma_b
            
            # Compute the posterior
            A = sigma_inv * (Lm_inv_Kmn @ Lm_inv_Kmn.T) + np.eye(m)
            L_A = np.linalg.cholesky(A)
            
            # Compute residuals
            res = Y_train - H_train @ b
            
            # Solve systems efficiently
            tmp = sigma_inv * np.linalg.solve(L_mm.T, (Lm_inv_Kmn @ res))
            mu = np.linalg.solve(L_A.T, np.linalg.solve(L_A, tmp))
            
            # Compute updated beta incorporating prior
            H_train_T = H_train.T
            B_inv = np.eye(H_train.shape[1]) * sigma_b_inv
            
            # Approximation of K_inv for VFE
            K_inv_approx = sigma_inv * np.eye(n_train) - \
                        sigma_inv**2 * Lm_inv_Kmn.T @ np.linalg.solve(A, Lm_inv_Kmn)
            
            beta = np.linalg.solve(
                B_inv + H_train_T @ K_inv_approx @ H_train,
                H_train_T @ K_inv_approx @ Y_train + B_inv @ b
            )
            
            # Final prediction
            pred = H_test @ beta + K_star_m @ np.linalg.solve(K_mm, mu)
        
        return pred

    def run(self, feat_train, feat_test, loc_train, loc_test, year_train, year_test,
            train_yield, model_weights, model_bias):
        """
        Run GP regression, using sparse approximation if specified
        """
        if self.use_sparse:
            print(f"Running Sparse GP with {self.num_inducing} inducing points...")
            return self.run_sparse(feat_train, feat_test, loc_train, loc_test, 
                                 year_train, year_test, train_yield, model_weights, model_bias)
        else:
            print("Running full GP...")
            return self.run_original(feat_train, feat_test, loc_train, loc_test,
                                    year_train, year_test, train_yield, model_weights, model_bias)
