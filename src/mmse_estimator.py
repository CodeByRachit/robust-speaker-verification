import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class PCAMMSEEstimator:
    """
    Minimum Mean Square Error (MMSE) estimator for vocal effort transfer vector
    in a PCA domain, strictly following Equations (4)-(10) of Lopez-Espejo et al. (2023).
    """
    def __init__(self, n_components_pca=16, n_components_gmm=8):
        self.L = n_components_pca
        self.K = n_components_gmm
        self.pca = PCA(n_components=self.L)
        self.gmm = GaussianMixture(
            n_components=self.K,
            covariance_type='full',
            reg_covar=1e-4,
            random_state=42
        )
        
        self.W_L = None       # shape (D, L) as in paper Eq. (4)
        self.weights = None   # shape (K,)
        self.mu_v = None      # shape (K, L)
        self.mu_y = None      # shape (K, L)
        self.diag_yy = None   # shape (K, L) - diagonal elements of Sigma_yy^(k)
        self.diag_vy = None   # shape (K, L) - diagonal elements of Sigma_vy^(k)
        
    def fit(self, X_normal, X_non_neutral):
        """
        Train the PCA transform W_L and the joint GMM p(z = (v, y)) offline.
        Args:
            X_normal (np.ndarray): Paired normal embeddings \tilde{x}, shape (N, D)
            X_non_neutral (np.ndarray): Paired non-neutral embeddings \tilde{y}, shape (N, D)
        """
        # Eq. (1): \tilde{v} = \tilde{y} - \tilde{x}
        V_tilde = X_non_neutral - X_normal
        
        # Fit PCA on normal and non-neutral embeddings to get principal eigenvectors W_L
        X_all = np.vstack((X_normal, X_non_neutral))
        self.pca.fit(X_all)
        
        # In sklearn, components_ has shape (L, D). In paper Eq. (4), W_L is (D, L).
        self.W_L = self.pca.components_.T  # shape (D, L)
        
        # Eq. (4): v = W_L^T \tilde{v} and y = W_L^T \tilde{y}
        V_pca = np.dot(V_tilde, self.W_L)        # shape (N, L)
        Y_pca = np.dot(X_non_neutral, self.W_L)  # shape (N, L)
        
        # Joint variable z = (v, y) in R^{2L}
        Z = np.hstack((V_pca, Y_pca))            # shape (N, 2L)
        
        # Fit K-component GMM
        self.gmm.fit(Z)
        
        self.weights = self.gmm.weights_
        mu_z = self.gmm.means_
        Sigma_z = self.gmm.covariances_
        
        # Eq. (6): Partition means and extract the LxL DIAGONAL sub-matrices
        self.mu_v = mu_z[:, :self.L]
        self.mu_y = mu_z[:, self.L:]
        
        # Extract diagonals of Sigma_vy^(k) and Sigma_yy^(k) as specified in Eq. (6)
        self.diag_vy = np.diagonal(Sigma_z[:, :self.L, self.L:], axis1=1, axis2=2)  # shape (K, L)
        self.diag_yy = np.diagonal(Sigma_z[:, self.L:, self.L:], axis1=1, axis2=2)  # shape (K, L)
        
    def compensate(self, y_tilde):
        """
        Perform online compensation for observed non-neutral embedding(s) \tilde{y}.
        Returns compensated normal embedding(s) \hat{\tilde{x}} = \tilde{y} - W_L \hat{v}.
        """
        single_input = (y_tilde.ndim == 1)
        if single_input:
            y_tilde = y_tilde.reshape(1, -1)
            
        N, D = y_tilde.shape
        x_hat_list = []
        
        for i in range(N):
            y_orig = y_tilde[i]  # shape (D,)
            
            # Eq. (4): y = W_L^T \tilde{y}
            y_pca = np.dot(y_orig, self.W_L)  # shape (L,)
            
            # Eq. (8): Compute posterior weights P(k|y) using diagonal covariance Sigma_yy^(k)
            log_p_y_given_k = np.zeros(self.K)
            for k in range(self.K):
                diff = y_pca - self.mu_y[k]
                var_yy = self.diag_yy[k]  # shape (L,)
                
                # Log Gaussian density for diagonal covariance matrix
                term = -0.5 * np.sum((diff ** 2) / var_yy)
                log_det = np.sum(np.log(var_yy))
                log_p_y_given_k[k] = term - 0.5 * log_det
                
            log_numerator = log_p_y_given_k + np.log(self.weights + 1e-12)
            max_log = np.max(log_numerator)
            p_k_given_y = np.exp(log_numerator - max_log) / np.sum(np.exp(log_numerator - max_log))
            
            # Eq. (7) & Eq. (9): Compute \hat{v} = \sum_{k=1}^K P(k|y) E(v|y, k)
            v_hat_pca = np.zeros(self.L)
            for k in range(self.K):
                diff = y_pca - self.mu_y[k]
                # Because Sigma_vy^(k) and Sigma_yy^(k) are diagonal LxL matrices (Eq. 6):
                # Sigma_vy^(k) * (Sigma_yy^(k))^-1 * (y - mu_y^(k)) simplifies to element-wise operations
                e_v_given_y_k = self.mu_v[k] + (self.diag_vy[k] / self.diag_yy[k]) * diff
                v_hat_pca += p_k_given_y[k] * e_v_given_y_k
                
            # Eq. (10): \hat{\tilde{v}} = W_L \hat{v} and \hat{\tilde{x}} = \tilde{y} - \hat{\tilde{v}}
            v_hat_orig = np.dot(self.W_L, v_hat_pca)  # shape (D,)
            x_hat_orig = y_orig - v_hat_orig
            x_hat_list.append(x_hat_orig)
            
        x_hat_array = np.vstack(x_hat_list)
        return x_hat_array[0] if single_input else x_hat_array