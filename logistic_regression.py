import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os
from sklearn.metrics import log_loss

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    
    X2 += np.array([distance, distance])
    y2 = np.ones(n_samples)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_regression(X, y):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    
    shift_distances = np.linspace(start, end, step_num)  # Range of shift distances
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}

    n_samples = step_num
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    epsilon = 1e-8

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)

        if abs(beta2) > epsilon:
            slope = -beta1 / beta2
            intercept = -beta0 / beta2
        else:
            slope = np.nan
            intercept = np.nan
            print(f"Warning: beta2 is zero or near zero at shift distance {distance}. Slope is undefined.")

        slope_list.append(slope)
        intercept_list.append(intercept)

        probs = model.predict_proba(X)
        loss = log_loss(y, probs)
        loss_list.append(loss)

        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0')
        plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1')

        if not np.isnan(slope) and not np.isnan(intercept):
            x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
            x1_values = np.linspace(x1_min, x1_max, 100)
            x2_values = slope * x1_values + intercept
            plt.plot(x1_values, x2_values, color='black', linestyle='--', linewidth=2, label='Decision Boundary')

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)  # Fading red for Class 1
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)  # Fading blue for Class 0
            if level == 0.7:
                try:
                    distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices, class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                    min_distance = np.min(distances)
                    margin_widths.append(min_distance)
                except IndexError:
                    min_distance = np.nan
                    margin_widths.append(min_distance)
                    print(f"Warning: Unable to calculate margin width at shift distance {distance}.")
        plt.title(f"Shift Distance = {distance}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")

        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\n"
        if not np.isnan(slope) and not np.isnan(intercept):
            equation_text += f"x2 = {slope:.2f} * x1 + {intercept:.2f}"
        else:
            equation_text += "Slope is undefined"
        margin_text = f"Margin Width: {min_distance:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=16, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=16, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if i == 1:
            plt.legend(loc='lower right', fontsize=16)

        sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    plt.figure(figsize=(18, 15))

    plt.subplot(3, 3, 1)
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")
    plt.plot(shift_distances, beta0_list, marker='o')

    plt.subplot(3, 3, 2)
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")
    plt.plot(shift_distances, beta1_list, marker='o')

    plt.subplot(3, 3, 3)
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")
    plt.plot(shift_distances, beta2_list, marker='o')

    plt.subplot(3, 3, 4)
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope (-Beta1/Beta2)")

    slope_array = np.array(slope_list)
    shift_array = np.array(shift_distances)

    valid_indices = ~np.isnan(slope_array) & ~np.isinf(slope_array)

    plt.plot(shift_array[valid_indices], slope_array[valid_indices], marker='o')

    plt.subplot(3, 3, 5)
    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept Ratio)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0 / Beta2")
    beta0_array = np.array(beta0_list)
    beta2_array = np.array(beta2_list)
    intercept_ratio = np.divide(-beta0_array, beta2_array, out=np.full_like(beta0_array, np.nan), where=beta2_array!=0)
    plt.plot(shift_distances, intercept_ratio, marker='o')

    plt.subplot(3, 3, 6)
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")
    plt.plot(shift_distances, loss_list, marker='o')

    plt.subplot(3, 3, 7)
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")
    plt.plot(shift_distances, margin_widths, marker='o')

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
