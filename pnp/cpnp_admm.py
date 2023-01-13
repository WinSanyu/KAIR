import torch

def irl1(f, u, v, b, sigma, lamb, irl1_iter_num, eps):
    return v

def admm(model, f, u, v, b, sigma, lamb, irl1_iter_num, eps):
    model_input = u / 255.
    u1 = model(model_input) * 255.
    b1 = b # self.mu
    v1 = irl1(f, u1, v, b1, sigma, lamb, irl1_iter_num, eps)
    return u1, v1, b1