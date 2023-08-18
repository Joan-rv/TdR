import numpy as np


class Optimitzador():
    def __init__(self):
        pass

    def actualitza(self):
        pass


def text_a_optimitzador(text, dimensions_sortida, dimensions_entrada):
    if text == 'cap':
        return Cap()
    elif text == 'adam':
        return Adam(dimensions_sortida, dimensions_entrada)
    else:
        raise Exception(f"Optimitzador {text} desconegut")


class Adam(Optimitzador):
    def __init__(self, dimensions_sortida, dimensions_entrada, beta1=0.1, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        valors_inicials = np.zeros((dimensions_sortida, dimensions_entrada)), np.zeros(
            (dimensions_sortida, 1))
        self.m_dW, self.m_db = valors_inicials
        self.v_dW, self.v_db = valors_inicials

    def actualitza(self, alfa, W, dW, b, db, iter):
        self.m_dW = self.beta1 * self.m_dW + (1 - self.beta1) * dW
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db
        self.v_dW = self.beta2 * self.v_dW + (1 - self.beta2) * dW**2
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * db**2

        beta1_elevat = self.beta1**iter
        beta2_elevat = self.beta2**iter
        m_corregit_dW = self.m_dW/(1 - beta1_elevat)
        m_corregit_db = self.m_db/(1 - beta1_elevat)
        v_corregit_dW = self.v_dW/(1 - beta2_elevat)
        v_corregit_db = self.v_db/(1 - beta2_elevat)

        W -= alfa * m_corregit_dW/(np.sqrt(v_corregit_dW) + self.epsilon)
        b -= alfa * m_corregit_db/(np.sqrt(v_corregit_db) + self.epsilon)

        return W, b


class Cap(Optimitzador):
    def __init__(self):
        pass

    def actualitza(self, alfa, W, dW, b, db, _):
        W -= alfa * dW
        b -= alfa * db
        return W, b
