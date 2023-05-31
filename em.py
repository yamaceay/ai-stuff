import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mvn

criterion = 'aic' # 'bic' or 'aic'

plt.style.use('ggplot')

np.set_printoptions(formatter={'all':lambda x: '%.3f' % x})

def em_gmm(xs, tol=0.01, max_iter=100):
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
    N = 1000 # size of samples
    LARGE_M = 25 # number of mixture components
    m_l, m_u = 2, 15 # range of mixture components to test
    
    # Define bounds for parameters
    mu_l, mu_u = -2, 2
    sigma_l, sigma_u = -2.5, 2.5
    
    # Define ground-truth parameters
    _mus = np.random.uniform(-2,2,(LARGE_M,2))
    _sigmas = np.random.uniform(sigma_l,sigma_u,(LARGE_M,2,2))
    _sigmas = np.array([np.dot(s.T, s) for s in _sigmas])
    _pis = np.random.random(LARGE_M)
    _pis /= _pis.sum()
    
    # Generate data
    xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*N))
        for pi, mu, sigma in zip(_pis, _mus, _sigmas)])
    
    scores = []
    for M in range(m_l, m_u):
        
        # Run EM
        ll1, pis1, mus1, sigmas1 = em_gmm(xs)
        
        # Calculate the score
        if criterion == 'bic':
            score = np.log(N)*3*M - 2*ll1
        elif criterion == 'aic':
            score = 2*3*M - 2*ll1
        else:
            raise ValueError("Invalid criterion")
            
        score /= N * LARGE_M
        scores.append((score, M))

        # Generate grid for plotting
        intervals = 101
        l, u = -8, 8
        ys = np.linspace(l, u, intervals)
        X, Y = np.meshgrid(ys, ys)
        _ys = np.vstack([X.ravel(), Y.ravel()]).T

        z = predict(_ys, pis1, mus1, sigmas1, intervals)

        # Plot the results
        ax = plt.subplot(111)
        plt.scatter(xs[:,0], xs[:,1], alpha=0.25)
        plt.contour(X, Y, z)
        plt.axis([l,u,l,u])
        plt.title(f"M = {M}, Score: {score:.2f}")
        ax.axes.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
    print(sorted(scores))