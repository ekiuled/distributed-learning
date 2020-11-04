import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ConsensusNode:
    def __init__(self, name: str,
                 weights: dict,
                 X_train, y_train, X_test, y_test,
                 stat_step=50, neighbors=None):
        self.alpha = 5e-4
        self.tau = 1e-4

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.name: str = name
        self.neighbors: dict = neighbors
        self.weights: dict = weights
        self.parameters: dict = dict()

        self.w = np.zeros((X_train.shape[1]))

        self.curr_iter: int = 0
        self.train_loss: int = 0

        self.stat_step: int = stat_step
        self.accuracy_list: list = []
        self.iter_list: list = []
        self.loss_list: list = []

    def _calc_accuracy(self):
        test_predictions = (self._sigmoid(self.X_test @ self.w) >= 0.5).astype(np.int) * 2 - 1
        return np.mean(test_predictions == self.y_test)

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_params(self):
        return self.w

    def show_graphs(self):
        fig, axs = plt.subplots(figsize=(20, 8), ncols=2)
        fig.suptitle(f'{self.name}', fontsize=24)
        fig.tight_layout(pad=4.0)
        sns.lineplot(x=self.iter_list, y=self.accuracy_list, ax=axs[0])
        axs[0].set_xlabel('Iteration', fontsize=16)
        axs[0].set_ylabel('Accuracy', fontsize=16)
        sns.lineplot(x=self.iter_list, y=self.loss_list, ax=axs[1])
        axs[1].set_xlabel('Iteration', fontsize=16)
        axs[1].set_ylabel('Loss', fontsize=16)

    def ask_params(self):
        self.parameters = {node_name: node.get_params()
                           for node_name, node in self.neighbors.items()}

    def update_params(self):
        self.w *= self.weights[self.name]

        for node_name, params in self.parameters.items():
            self.w += params * self.weights[node_name]

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit_step(self, epoch):
        self.curr_iter += 1

        grad = -np.array([
            np.dot(self.y_train * self._sigmoid(- self.y_train * (self.X_train @ self.w)), self.X_train[:, j])
            for j in range(self.X_train.shape[1])
        ]) / self.X_train.shape[0] + self.tau * self.w
        self.w -= self.alpha * grad

        self.train_loss = self.tau / 2 * np.sum(self.w**2) + -np.mean(np.log(self._sigmoid(self.y_train * (self.X_train @ self.w))))

        # Save stats
        if self.curr_iter % self.stat_step == 0:
            self.iter_list.append(self.curr_iter)
            self.loss_list.append(float(self.train_loss))
            self.train_loss = 0
            self.accuracy_list.append(float(self._calc_accuracy()))
            # print(f"Epoch: {epoch}, Step: {self.curr_iter}, Node {self.name}:"
            #       f" accuracy {self.accuracy_list[-1]:.2f}, loss {self.loss_list[-1]:.2f}")
