import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def sample_auxiliary_points(P: np.ndarray, n_points: int, method: str = 'scott'):
    assert method in ['scott', 'silverman']

    success = False
    while not success:
        try:
            kde = stats.gaussian_kde(P.T, bw_method=method)
            success = True
        except Exception as e:
            additional_point = np.random.choice(np.arange(P.shape[1]))
            additional_point += np.random.normal(0, 0.001, 2)
            P = np.hstack([P, additional_point])

    samples = kde.resample(size=n_points)
    return samples.T


if __name__ == '__main__':
    n_samples = 500
    
    mean1, cov1 = [2, 2], [[1, 0.5], [0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 2)

    mean2, cov2 = [-2, -2], [[1, -0.8], [-0.8, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 4)
    
    mean3, cov3 = [4, -3], [[0.5, 0], [0, 0.5]]
    data3 = np.random.multivariate_normal(mean3, cov3, n_samples // 4)
    
    data_original = np.vstack([data1, data2, data3]).T

    data_resampled = sample_auxiliary_points(data_original.T, 2000).T

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    
    axes[0].scatter(data_original[0], data_original[1], s=10, alpha=0.5, c='blue', label='Source Data')
    axes[0].set_title("Source", fontsize=14)
    axes[1].scatter(data_resampled[0], data_resampled[1], s=10, alpha=0.5, c='red', label='Resampled (KDE)')
    axes[1].set_title(f"Samples(N={data_resampled.shape[1]})", fontsize=14)
    
    xmin, xmax = -6, 8
    ymin, ymax = -6, 6
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)

    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)


    kde = stats.gaussian_kde(data_original, bw_method='scott')
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kde(positions).T, xx.shape)
    
    cf = axes[2].contourf(xx, yy, f, cmap='viridis', levels=20)
    axes[2].set_title("Inferenced density", fontsize=14)
    plt.colorbar(cf, ax=axes[2])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()