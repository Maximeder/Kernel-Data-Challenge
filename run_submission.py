import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
from sklearn.model_selection import train_test_split


def image_construction(image_vect, mode='rgb'):
    ''' transforms a (3072, ) vector into an (32, 32, 3) RGB image '''
    image = image_vect.copy()
    image = np.transpose(np.reshape(image, (3, 32, -1)), (1, 2, 0))

    r_min, r_max = image[:, :, 0].min(), image[:, :, 0].max()
    g_min, g_max = image[:, :, 1].min(), image[:, :, 1].max()
    b_min, b_max = image[:, :, 2].min(), image[:, :, 2].max()

    image[:, :, 0] = (image[:, :, 0] - r_min) / (r_max - r_min)
    image[:, :, 1] = (image[:, :, 1] - g_min) / (g_max - g_min)
    image[:, :, 2] = (image[:, :, 2] - b_min) / (b_max - b_min)

    if mode == 'rgb':
        return image
    if mode == 'gray':
        return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]


class linear:
    def __init__(self):
        self = self

    def kernel(self, X1, X2):
        return X1 @ X2.T


class poly:
    def __init__(self, d=3, c=0):
        self.d = d
        self.c = c

    def kernel(self, X1, X2):
        return (X1 @ X2.T + self.c) ** self.d


class rbf:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def kernel(self, X1, X2):
        n, p = X1.shape
        m, _ = X2.shape
        X1 = X1.reshape(n, 1, p, 1)
        X2 = X2.reshape(1, m, p, 1)
        A = ((X1 - X2).transpose((0, 1, 3, 2)) @ (X1 - X2)).reshape(n, m)
        return np.exp(-A / (2 * self.sigma ** 2))


class KernelSVC:

    def __init__(self, C, kernel, epsilon=1e-8):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support_vectors = None
        self.weight = None
        self.b = None
        self.epsilon = epsilon

    def fit(self, X, y):
        n = len(y)
        K = self.kernel(X, X)

        def loss(alpha):
            return -np.sum(alpha) + 0.5 * (alpha * y) @ K @ (alpha * y)  # opposite of the loss -> use of a minimizer

        def grad_loss(alpha):
            return -np.ones(n) + np.diag(y) @ K @ np.diag(y) @ alpha

        eq_constraint = lambda alpha: (-np.sum(y * alpha)).reshape(1, 1)
        jac_eq_constraint = lambda alpha: -y

        ineq_constraint = lambda alpha: (np.hstack([self.C * np.ones(n) - alpha, alpha])).reshape(2 * n, 1)
        jac_ineq_constraint = lambda alpha: np.vstack((-np.eye(n), np.eye(n)))

        constraints = ({'type': 'eq', 'fun': eq_constraint, 'jac': jac_eq_constraint},
                       {'type': 'ineq', 'fun': ineq_constraint, 'jac': jac_ineq_constraint})

        self.alpha = minimize(fun=lambda alpha: loss(alpha), x0=np.ones(n), method='SLSQP',
                              jac=lambda alpha: grad_loss(alpha), constraints=constraints).x

        margin_vectors_mask = np.logical_and(self.alpha > self.epsilon, self.alpha < self.C - self.epsilon)
        self.b = np.mean((y - (y * self.alpha) @ K)[margin_vectors_mask])

        support_vectors_mask = (self.alpha > self.epsilon)
        self.support_vectors = X[support_vectors_mask]
        self.weight = y[support_vectors_mask] * self.alpha[support_vectors_mask]  # weight vector used for
        # prediction

    def predict(self, X_pred):
        soft_prediction = self.b + self.weight @ self.kernel(self.support_vectors, X_pred)
        return np.sign(soft_prediction)


class KernelSVC_1V1:

    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.n_class = 0
        self.binary_classifiers = {}

    def fit(self, X, y):
        self.n_class = len(np.unique(y))

        for class_i in range(self.n_class):
            for class_j in range(class_i + 1, self.n_class):
                mask = np.logical_or(y == class_i, y == class_j)
                X_train = X[mask]
                y_train = np.where(y[mask] == class_i, 1, -1)
                clf = KernelSVC(C=self.C, kernel=self.kernel)
                clf.fit(X_train, y_train)

                self.binary_classifiers[(class_i, class_j)] = clf

    def predict(self, X_pred):
        classes_scores = np.zeros(shape=(X_pred.shape[0], self.n_class))

        for class_i in range(self.n_class):
            for class_j in range(class_i + 1, self.n_class):
                clf = self.binary_classifiers[(class_i, class_j)]
                duel_i_j = clf.predict(X_pred)

                classes_scores[duel_i_j == 1, class_i] += 1  # +1 in class_i in the rows where class_i won
                classes_scores[duel_i_j == -1, class_j] += 1  # +1 in class_j in the rows where class_j won

        return np.argmax(classes_scores, axis=1)


def gray_image_gradient(image):
    grad_x = np.zeros(shape=(32, 32))
    grad_y = np.zeros(shape=(32, 32))

    # computation of the gradient along the x and the y axis
    grad_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
    grad_x[:, 0] = image[:, 1] - image[:, 0]
    grad_x[:, -1] = image[:, -1] - image[:, -2]

    grad_y[1:-1] = image[2:, :] - image[:-2, :]
    grad_y[0, :] = image[1, :] - image[0, :]
    grad_y[-1, :] = image[-1, :] - image[-2, :]

    return grad_x, grad_y


def magnitude_orientation(grad_x, grad_y):
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = (180 / np.pi) * np.arctan(grad_y / grad_x) + 90

    return magnitude, orientation


def cell_histogram(cell_magnitude, cell_orientation, hist_bins):
    # Builds the histogram of orientations on one given cell
    cell_hist = np.zeros(len(hist_bins))

    cell_hist[0] = cell_magnitude[cell_orientation < hist_bins[0]].sum()
    for i in range(1, len(hist_bins)):
        v_min, v_max = hist_bins[i - 1], hist_bins[i]
        cell_hist[i] = cell_magnitude[np.logical_and(cell_orientation >= v_min, cell_orientation < v_max)].sum()

    return cell_hist


def hog(image, cell_size=(8, 8), n_hist_bins=9):
    c = cell_size[0]
    n = 32 // c

    hist_bins = np.linspace(0, 180, n_hist_bins + 1)[1:]

    grad_x, grad_y = gray_image_gradient(image)
    magnitude, orientation = magnitude_orientation(grad_x, grad_y)

    hog_list = []

    for i, j in product(np.arange(n), np.arange(n)):
        cell_magnitude = magnitude[c * i:c * (i + 1), c * j:c * (j + 1)]
        cell_orientation = orientation[c * i:c * (i + 1), c * j:c * (j + 1)]

        cell_hist = cell_histogram(cell_magnitude, cell_orientation, hist_bins)

        hog_list.append(cell_hist)

    return np.hstack(hog_list)


def hog_transform(X, cell_size=(8, 8), n_hist_bins=9):
    X_transformed = []
    for x in X:
        image = image_construction(x, mode='gray')
        hog_features = hog(image, cell_size, n_hist_bins)
        X_transformed.append(hog_features)

    return np.vstack(X_transformed)

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()

Xtr_t = hog_transform(Xtr)
Xte_t = hog_transform(Xte)

# # #  Enter the chosen parameter :
kernel = rbf(sigma=5).kernel
clf = KernelSVC_1V1(C=1000, kernel=kernel)
# # #

clf.fit(Xtr_t, Ytr)
prediction = clf.predict(Xte_t)

prediction_submit = {'Prediction' : prediction}
dataframe = pd.DataFrame(prediction_submit)
dataframe.index += 1
dataframe.head()
dataframe.to_csv('submission.csv',index_label='Id')

