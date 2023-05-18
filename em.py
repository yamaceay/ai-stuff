import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mvn
plt.style.use('ggplot')

np.set_printoptions(formatter={'all':lambda x: '%.3f' % x})

def em_gmm_orig(xs, tol=0.01, max_iter=100):
    """Original version of EM for GMM"""
    
    # Estimate parameters
    mus = np.random.random((M,2))
    sigmas = np.array([np.eye(2)] * M)
    pis = np.random.random(M)
    pis /= pis.sum()
    
    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for _ in range(max_iter):
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
        ws /= ws.sum(0)

        # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]
            mus[j] /= ws[j, :].sum()

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mus[j], (2,1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= ws[j,:].sum()

        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas

def em_gmm_vect(xs, tol=0.01, max_iter=100):
    """Vectorized version of EM for GMM"""

    # Estimate parameters
    mus = np.random.random((M,2))
    sigmas = np.array([np.eye(2)] * M)
    pis = np.random.random(M)
    pis /= pis.sum()

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for _ in range(max_iter):
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = ws.sum(axis=1)
        pis /= n

        mus = np.dot(ws, xs)
        mus /= ws.sum(1)[:, None]

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            ys = xs - mus[j, :]
            sigmas[j] = (ws[j,:,None,None] * np.matmul(ys[:,:,None], ys[:,None,:])).sum(axis=0)
        sigmas /= ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*mvn(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas

def em_gmm_eins(xs, tol=0.01, max_iter=100):
    """Einstein summation version of EM for GMM"""
    
    # Estimate parameters
    mus = np.random.random((M,2))
    sigmas = np.array([np.eye(2)] * M)
    pis = np.random.random(M)
    pis /= pis.sum()
    
    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for _ in range(max_iter):
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j, (pi, mu, sigma) in enumerate(zip(pis, mus, sigmas)):
            ws[j, :] = pi * mvn(mu, sigma).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = np.einsum('kn->k', ws)/n
        mus = np.einsum('kn,np -> kp', ws, xs)/ws.sum(1)[:, None]
        sigmas = np.einsum('kn,knp,knq -> kpq', ws,
            xs-mus[:,None,:], xs-mus[:,None,:])/ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*mvn(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas

def predict(_ys: np.ndarray, pis1, mus1, sigmas1, intervals) -> np.ndarray:
    z = np.zeros(len(_ys))
    for pi, mu, sigma in zip(pis1, mus1, sigmas1):
        z += pi*mvn(mu, sigma).pdf(_ys)
    z = z.reshape((intervals, intervals))
    
    return z

if __name__ == "__main__":    
    M = 5 # number of different classes
    N = 200 # size of samples
    
    mu_l, mu_u = -2, 2
    sigma_l, sigma_u = -2.5, 2.5
    
    # Define ground-truth parameters
    _mus = np.random.uniform(mu_l,mu_u,(M,2))
    _sigmas = np.random.uniform(sigma_l,sigma_u,(M,2,2))
    _sigmas = np.array([np.dot(s.T, s) for s in _sigmas])
    _pis = np.random.random(M)
    _pis /= _pis.sum()
    
    for i, (_pi, _mu, _sigma) in enumerate(zip(_pis, _mus, _sigmas)):
        print(f"p_{i}: {_pi}")
        print(f"mean_{i}: {_mu}")
        print(f"cov_{i}: {_sigma}")
    
    # Generate data
    xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*N))
                        for pi, mu, sigma in zip(_pis, _mus, _sigmas)])
    
    # Run EM
    ll1, pis1, mus1, sigmas1 = em_gmm_eins(xs)

    # Plot results
    intervals = 101
    l, u = -8, 8
    ys = np.linspace(l, u, intervals)
    X, Y = np.meshgrid(ys, ys)
    _ys = np.vstack([X.ravel(), Y.ravel()]).T

    z = predict(_ys, pis1, mus1, sigmas1, intervals)

    ax = plt.subplot(111)
    plt.scatter(xs[:,0], xs[:,1], alpha=0.25)
    plt.contour(X, Y, z)
    plt.axis([l,u,l,u])
    ax.axes.set_aspect('equal')
    plt.tight_layout()
    plt.show()