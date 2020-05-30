
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""This module provides functions for the computation of stationary
vectors of stochastic matrices.

Matrices are represented by scipy.sparse matrices throughout this module.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse.linalg

from scipy.sparse import eye
from scipy.sparse.linalg import factorized
EPS = np.finfo(np.float64).eps


def backward_iteration(A, mu, x0, tol=1e-14, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A : (N, N) scipy.sparse matrix
        Matrix for which eigenvector is desired
    mu : float
        Approximate eigenvalue for desired eigenvector
    x0 : (N, ) ndarray
        Initial guess for eigenvector
    tol : float
        Tolerace parameter for termination of iteration

    Returns
    -------
    x : (N, ) ndarray
        Eigenvector to approximate eigenvalue mu

    """
    T = A - mu * eye(A.shape[0], A.shape[0])
    T = T.tocsc()
    """Prefactor T and return a function for solution"""
    solve = factorized(T)
    """Starting iterate with ||y_0||=1"""
    r0 = 1.0 / np.linalg.norm(x0)
    y0 = x0 * r0
    """Local variables for inverse iteration"""
    y = 1.0 * y0
    r = 1.0 * r0
    N = 0
    for i in range(maxiter):
        x = solve(y)
        r = 1.0 / np.linalg.norm(x)
        y = x * r
        if r <= tol:
            return y
    msg = "Failed to converge after %d iterations, residuum is %e" % (maxiter, r)
    raise RuntimeError(msg)


def stationary_distribution_from_backward_iteration(P, eps=1e-15):
    r"""Fast computation of the stationary vector using backward
    iteration.

    Parameters
    ----------
    P : (M, M) scipy.sparse matrix
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.

    Returns
    -------
    pi : (M,) ndarray
        Stationary vector

    """
    A = P.transpose()
    mu = 1.0 - eps
    x0 = np.ones(P.shape[0])
    y = backward_iteration(A, mu, x0)
    pi = y / y.sum()
    return pi


def stationary_distribution_from_eigenvector(T, ncv=None):
    r"""Compute stationary distribution of stochastic matrix T.

    The stationary distribution is the left eigenvector corresponding to the 1
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """

    # get the top two eigenvalues and vecs so we can check for irreducibility
    vals, vecs = scipy.sparse.linalg.eigs(T.transpose(), k=2, which='LR', ncv=ncv)

    # check for irreducibility
    if np.allclose(vals, 1, rtol=1e2*EPS, atol=1e2*EPS):
        raise ValueError('This matrix is reducible')

    # sort by real part and take the top one
    p = np.argsort(vals.real)[::-1]
    vecs = vecs[:, p]
    top_vec = vecs[:, 0]

    # check for imaginary component
    imaginary_component = top_vec.imag
    if not np.allclose(imaginary_component, 0, rtol=EPS, atol=EPS):
        raise ValueError('Top eigenvector has imaginary component')
    top_vec = top_vec.real

    # check the sign structure
    if not (top_vec > -1e4*EPS).all() and not (top_vec < 1e4*EPS).all():
        raise ValueError('Top eigenvector has both positive and negative entries')
    top_vec = np.abs(top_vec)

    # normalize to 1 and return
    return top_vec / np.sum(top_vec)


def stationary_distribution(T):
    r"""Compute stationary distribution of stochastic matrix T.

    Chooses the fastest applicable algorithm automatically

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    mu = stationary_distribution_from_eigenvector(T)

    return mu
