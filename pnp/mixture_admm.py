import torch
from math import log, sqrt

# def subproblem_L(Lambda1, Lambda2, beta1, beta2, Y, X2, S2, L2):
#     Lambda1 += beta1 * (Y - X2 - S2)
#     Lambda2 += beta2 * (X2 - L2)
#     return Lambda1, Lambda2

# def subproblem_S(Y, X2, Lambda1, beta1, lamb):
#     soft = _get_soft(lamb/beta1)
#     S2 = soft(Y - X2 + Lambda1/beta1)
#     return S2

# def subproblem_L(X2, Lambda2, beta2, sigma):
#     model = lambda x: x # TODO: choice denoisor
#     L2 = model(X2 + Lambda2/beta2)
#     return L2

# def _get_Laplace(x, xi, alpha):
#     return (x - xi)**2 - 4*(alpha - x*xi)

# def _phi(alpha, y, xi, x):
#     return alpha*torch.log(y + xi) + (y - x)**2 / 2

# def _get_argminphi(x, Laplace, xi, alpha):
#     y1 = torch.zeros_like(x)
#     phi_y1 = _phi(alpha, y1, xi, x)
#     y2 = (x - xi + torch.sqrt(Laplace)) / 2
#     phi_y2 = _phi(alpha, y2, xi, x)

#     y = torch.zeros_like(x)
#     y[phi_y1 > phi_y2] = y2[phi_y1 > phi_y2]
#     return y

# def subproblem_X(Y, S1, L1, Lambda1, Lambda2, beta1, beta2, xi):
#     A = Y - S1 + Lambda1/beta1
#     B = L1 - Lambda2/beta2
#     alpha = 1 / (beta1 + beta2)
#     G = (beta1*A + beta2*B) / (beta1 + beta2)

#     # G = U diag(Sigma) V'
#     U, Sigma, V = torch.svd(G)
#     # print(torch.dist(G, torch.matmul(torch.matmul(U, torch.diag_embed(Sigma)), V.mT)))
#     Laplace = _get_Laplace(Sigma, xi, alpha)
#     T = torch.zeros_like(Laplace)
#     T[Laplace > 0] = _get_argminphi(Sigma[Laplace > 0], Laplace[Laplace > 0], xi, alpha)
#     UTVh = torch.matmul(torch.matmul(U, torch.diag_embed(T)), V.mT)
#     return UTVh

def _get_soft(tao):
    return lambda x: torch.nn.functional.relu(torch.abs(x) - tao) * torch.sgn(x)

def subproblem_sigma(y, x, s, N):
    return torch.sqrt(torch.norm(y - x - s,'fro') / N)

def subproblem_x(beta, sigma, z, gamma, y, s):
    return (
        (beta * sigma**2 * z + sigma**2 * gamma + y - s) /
        (1 + beta*sigma**2)
    )

def subproblem_gamma(gamma, beta, x1, z1):
    gamma = gamma + beta * (x1 - z1)
    return gamma

def subproblem_S(y, mu, eps=1e-8):
    n = y.shape[-1]
    theta0 = torch.sqrt(torch.sum(y**2, -1)/n)

    alpha = y / (theta0 + eps)
    a = alpha**2
    b = -2*alpha*y
    c = 4*mu

    tmp1 = -b / (4*a + eps)
    tmp2 = b**2 / (16 * a**2) - c/(2*a)

    idx = tmp2 >= 0
    tmp1[idx == 0] = 0
    tmp2[idx == 0] = 0

    t1 = tmp1 + torch.sqrt(tmp2)
    t2 = tmp1 - torch.sqrt(tmp2)

    f0 = c * log(eps)
    f1 = a * t1**2 + b * t1 + c*torch.log(t1 + eps)
    f2 = a * t2**2 + b * t2 + c*torch.log(t2 + eps)

    ind = f2 < f1
    f1[ind] = f2[ind]
    t1[ind] = t2[ind]
    ind = f0 < f1
    t1[ind] = 0
    theta = t1

    aa = y / (theta + eps)
    thr = 2*sqrt(2)*mu / (theta**2 + eps)
    alpha = _get_soft(thr)(aa)
    S = theta*alpha
    return S

def subproblem_z(x1, gamma, beta, eta):
    denoisor = lambda x, y: x # TODO using network
    denoisor(x1 + gamma/beta, sqrt(eta/beta))