from typing import Tuple
import numpy as np


class MatrixFactorization:
    def __init__(
        self,
        R: np.ndarray,
        k: int,
        learning_rate: float,
        reg_param: float,
        epochs: int,
        verbose: bool = False,
    ):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = R
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """
        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init bias
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        self._training_process = []
        for epoch in range(self._epochs):
            xi, yi = self._R.nonzero()
            for i, j in zip(xi, yi):
                self.gradent_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            if self._verbose and ((epoch + 1) % 10 == 0):
                print(f"Iteration: {epoch + 1} ; cost = {cost}")

    def cost(self) -> float:
        """
        compute root mean square error
        :return: rmse cost
        """
        xi, yi = self._R.nonzero()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
        return np.sqrt(cost / len(xi))

    def gradient(
        self, error: np.ndarray, i: int, j: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq

    def gradient_descent(self, i: int, j: int, rating: np.ndarray):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq

    def get_prediction(self, i: int, j: int) -> np.ndarray:
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return (
            self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)
        )

    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return (
            self._b
            + self._b_P[:, np.newaxis]
            + self._b_Q[
                np.newaxis :,
            ]
            + self._P.dot(self._Q.T)
        )
