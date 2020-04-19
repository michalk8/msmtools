
# This file is part of MSMTools.
#
# Copyright (c) 2020 Bernhard Reuter, Susanna Roeblitz and Marcus Weber, 
# Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
# --------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------
# Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
# Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
# Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
# Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079
# ----------------------------------------------------------------
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

'''
@author: Bernhard Reuter, Marcus Weber, Susanna Roeblitz

'''

import warnings
import numpy as np
from scipy.sparse import issparse
import math

# Machine double floating precision:
eps = np.finfo(np.float64).eps


def _find_twoblocks(R):
    r"""
    This function checks the sorted part of the Schurform `R` for 2x2-blocks. 
    If a 2x2-block (corresponding to two complex conjugate eigenvalues, that MUST NOT be splitted) 
    at positions (``rr_i;i``, ``rr_i;i+1``, ``rr_i+1;i``, ``rr_i+1;i+1``) is found, the row-index ``i`
    of the first row of the 2x2-block is identified as invalid row-index and ``n_cluster = i+1``
    is excluded from the array of valid cluster numbers that is returned by this function.
    
    Parameters
    ----------
    R : ndarray (n,n)
        (Partially) sorted real Schur matrix of
        :math:`\tilde{P} = \mathtt{diag}(\sqrt{\eta}) P \mathtt{diag}(1.0. / \sqrt{eta})`
        such that :math:`\tilde{P} Q = Q R` with the (partially) sorted matrix 
        of Schur vectors :math:`Q` holds.
        
    Returns
    -------
    validclusters : ndarray (l,)
        Array of valid cluster numbers.
    """
    
    badindices = np.asarray(np.abs(np.diag(R, -1)) > 1000 * eps).nonzero()[0]
    validclusters = np.setdiff1d(np.arange(R.shape[0] + 1), badindices + 1)
    
    return validclusters
  
  
def _gram_schmidt_mod(X, eta):
    r"""
    Function to :math:`\eta`-orthonormalize Schur vectors - modified numerically stable version.
    
    Parameters
    ----------
    X : ndarray (n,m)
        Matrix consisting columnwise of the ``m`` dominant Schur vectors of 
        :math:`\tilde{P} = \mathtt{diag}(\sqrt{\eta}) P \mathtt{diag}(1.0. / \sqrt{eta})`.
        
    eta : ndarray (n,) 
        Input (initial) distribution of states.
        
    Returns
    -------
    Q : ndarray (n,m)
        Matrix with the orthonormalized ``m`` dominant Schur vectors of :math:`\tilde{P}`.
        The elements of the first column are constantly equal :math:`\sqrt{eta}`.
    
    """
    from scipy.linalg import subspace_angles
    
    # Keep copy of the original (Schur) vectors for later sanity check.
    Xc = np.copy(X)
    
    # Initialize matrices.
    n, m = X.shape
    Q = np.zeros((n,m))
    R = np.zeros((m,m))
    
    # Search for the constant (Schur) vector, if explicitly present.
    max_i = 0
    for i in range(m):
        vsum = np.sum(X[:,i])
        dummy = ( np.ones(X[:,i].shape) * (vsum / n) )
        if np.allclose(X[:,i], dummy, rtol=1e6*eps, atol=1e6*eps ):  
            max_i = i
        
    # Shift non-constant first (Schur) vector to the right.
    X[:,max_i] = X[:, 0]
    # Set first (Schur) vector equal sqrt(eta) (In _do_schur() the Q-matrix, orthogonalized by 
    # _gram_schmidt_mod(), will be multiplied with 1.0./sqrt(eta) - so the first (Schur) vector will 
    # become the unit vector 1!).
    X[:, 0] = np.sqrt(eta)
    # Raise, if the subspace changed! TODO: Mb test rank instead?
    if not ( subspace_angles(X, Xc)[0] < 1e8 * eps ): 
        raise ValueError("The subspace of Q derived by shifting a non-constant first (Schur)vector "
                         "to the right and setting the first (Schur) vector equal sqrt(eta) doesn't "
                         "match the subspace of the original Q!")
    
    # eta-orthonormalization
    for j in range(m):
        v = X[:,j] ;
        for i in range(j):
            R[i,j] = np.dot(Q[:,i].conj(), v)
            v = v - np.dot(R[i,j], Q[:,i])
        R[j,j] = np.linalg.norm(v) ;
        Q[:,j] = np.true_divide(v, R[j,j])

    # Raise, if the subspace changed! TODO: Mb test rank instead?
    if not ( subspace_angles(Q, Xc)[0] < 1e8 * eps ):
        raise ValueError("The subspace of Q derived by eta-orthogonalization doesn't match the "
                         + "subspace of the original Q!")
    # Raise, if the (Schur)vectors aren't orthogonal!
    if not np.allclose(Q.conj().T.dot(Q), np.eye(Q.shape[1]), rtol=1e6*eps, atol=1e6*eps):
        raise ValueError("(Schur)vectors appear to not be orthogonal!")
    
    return Q


def _do_schur(P, eta, m):
    r"""
    This function performs a Schur decomposition of the (n,n) transition matrix `P`, with due regard 
    to the input (initial) distribution of states `eta` (which can be the stationary distribution ``pi``,
    if a reversible matrix `P` is considered, or some initial (even arbitrarily choosen, e.g., uniform) 
    or average distribution of the `m` states, if a nonreversible `P` is evaluated). 
    Afterwards the Schur form and Schur vector matrix are sorted (sorting the `m` dominant (largest) 
    eigenvalues to the top left of the Schur form in descending order and correspondingly sorting 
    the associated Schur vectors to the left of the Schur vector matrix).
    Only the top left (m,m) part of the sorted Schur form and the associated left (n,m) part
    of the correspondingly sorted Schur vector matrix are returned.
    
    Parameters
    ----------
    P : ndarray (n,n)          
        Row-stochastic transition matrix.
        
    eta : ndarray (n,)         
        Input (initial) distribution of states.
        
    m : integer           
        Number of states or clusters, corresponding to the `m` dominant (largest) eigenvalues:
        
    Returns
    -------
    X : ndarray (n,m)
        Matrix containing the ordered `m` dominant Schur vectors columnwise.
    R : ndarray (m,m)
        The ordered top left Schur form.
    
    """
    
    from scipy.linalg import schur
    from scipy.linalg import subspace_angles
    from msmtools.util.sort_real_schur import sort_real_schur
    
    # Exeptions
    N1 = P.shape[0]
    N2 = P.shape[1]
    if m < 0:
        raise ValueError("The number of clusters/states is not supposed to be negative!")
    if not (N1==N2):
        raise ValueError("P matrix isn't quadratic!")
    if not (eta.shape[0]==N1):
        raise ValueError("eta vector length doesn't match with the shape of P!")
    if not np.allclose(np.sum(P,1), np.ones(N1), rtol=eps, atol=eps):
        raise ValueError("Not all rows of P sum up to one (within numerical precision)! "
                         "P must be a row-stochastic matrix!")
    if not np.all(eta > eps):
        raise ValueError("Not all elements of eta are > 0 (within numerical precision)!")

    # Weight the stochastic matrix P by the input (initial) distribution eta.
    P_bar = np.diag(np.sqrt(eta)).dot(P).dot(np.diag(1./np.sqrt(eta)))

    # Make a Schur decomposition of P_bar.
    R, Q = schur(P_bar,output='real') #TODO: 1. Use sort keyword of schur instead of sort_real_schur? 2. Implement Krylov-Schur (sorted partial Schur decomposition)

    # Sort the Schur matrix and vectors.
    Q, R, ap = sort_real_schur(Q, R, z=np.inf, b=m)
    # Warnings
    if np.any(np.array(ap) > 1.0):
        warnings.warn("Reordering of Schur matrix was inaccurate!")
    if m - 1 not in _find_twoblocks(R):
        warnings.warn("Coarse-graining with " + str(m) + " states cuts through a block of "
                      + "complex conjugate eigenvalues in the Schur form. The result will "
                      + "be of questionable meaning. "
                      + "Please increase/decrease number of states by one.")
        
    # Since the Schur form R and Schur vectors are only partially
    # sorted, one doesn't need the whole R and Schur vector matrix Q.
    # Take only the sorted Schur form and the vectors belonging to it.
    Q = Q[:, 0:m]
    R = R[0:m, 0:m]

    # Orthonormalize the sorted Schur vectors Q via modified Gram-Schmidt-orthonormalization
    Q = _gram_schmidt_mod(Q, eta)
         
    # Transform the orthonormalized Schur vectors of P_bar back to orthonormalized Schur vectors X of P.
    X = np.diag(1./np.sqrt(eta)).dot(Q)
    if not X.shape[0] == N1:
        raise ValueError("The number of rows n=%d of the Schur vector matrix X doesn't match those (n=%d) of P!" 
                         % (X.shape[0], P.shape[0]))
    # Raise, if the (Schur)vectors aren't D-orthogonal (don't fullfill the orthogonality condition)!
    if not np.allclose(X.conj().T.dot(np.diag(eta)).dot(X), np.eye(X.shape[1]), rtol=1e6*eps, atol=1e6*eps):
        raise ValueError("Schur vectors appear to not be D-orthogonal!")
    # Raise, if X doesn't fullfill the invariant subspace condition!
    if not ( subspace_angles(np.dot(P, X), np.dot(X, R))[0] < 1e8 * eps ):
        raise ValueError("X doesn't span a invariant subspace of P!")
    # Raise, if the first column X[:,0] of the Schur vector matrix isn't constantly equal 1!
    if not np.allclose(X[:,0], np.ones(X[:,0].shape), rtol=1e6*eps, atol=1e6*eps):
        raise ValueError("The first column X[:,0] of the Schur vector matrix isn't constantly equal 1!")
                  
    return X, R
  
  
def _objective(alpha, X):
    r"""
    Compute objective function value.
    
    Parameters
    ----------
    alpha : ndarray ((m-1)^2,)
        Vector containing the flattened croped rotation matrix ``rot_matrix[1:,1:]``.
        
    X : ndarray (n,m)
        A matrix with m sorted Schur vectors in the columns. The constant Schur vector should be first.
        
    Returns
    -------
    optval : float (double)
        Current value of the objective function :math:`f = m - trace(S)` (Eq. 16 from [1]_).
        
    References
    ----------
    .. [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6
    
    """
    # Dimensions.
    n, m = X.shape
    k = m - 1   
    
    # Initialize rotation matrix.
    rot_matrix = np.zeros((m, m))
    
    # Sanity checks.
    if not (alpha.shape[0] == k**2):
        raise ValueError("The shape of alpha doesn't match with the shape of X: "
                         + "It is not a ((" + str(m) + "-1)^2,)-vector, but of dimension " 
                         + str(alpha.shape) + ". X is of shape " + str(X.shape) + ".")
    
    # Now reshape alpha into a (k,k)-matrix.
    rot_crop_matrix = np.reshape(alpha, (k, k))
    
    # Complete rot_mat to meet constraints (positivity, partition of unity).
    rot_matrix[1:,1:] = rot_crop_matrix
    rot_matrix = _fill_matrix(rot_matrix, X)

    # Compute value of the objective function.
    # from Matlab: optval = m - trace( diag(1 ./ A(1,:)) * (A' * A) )
    optval = m - np.trace( np.diag(np.true_divide(1.0, rot_matrix[0, :])).dot(rot_matrix.conj().T.dot(rot_matrix)) )
    
    # Attention: Our definition of the objective function seems to differ from those used in MSMTools pcca.py!
    # They seem to use -result (from susanna_func() below in their _opt_soft()) for optimization in fmin, 
    # while one should use (k - result) - maybe, because they don't use optval to find the optimal number of
    # clusters (with the most crisp decomposition of the state space)...
    #-----------------------------------------------------------------------------------------
    ## Susanna Roeblitz' target function for optimization
    #def susanna_func(rot_crop_vec, eigvectors):
    #    # reshape into matrix
    #    rot_crop_matrix = np.reshape(rot_crop_vec, (x, y))
    #    # fill matrix
    #    rot_matrix = _fill_matrix(rot_crop_matrix, eigvectors)
    #
    #    result = 0
    #    for i in range(0, n_clusters):
    #        for j in range(0, n_clusters):
    #            result += np.power(rot_matrix[j, i], 2) / rot_matrix[0, i]
    #    return -result
    # 
    #from scipy.optimize import fmin
    #
    #rot_crop_vec_opt = fmin(susanna_func, rot_crop_vec, args=(eigvectors,), disp=False)
    #------------------------------------------------------------------------------------------
    
    return  optval
  
  
def _initialize_rot_matrix(X):
    r"""
    Initialize the rotation (m,m)-matrix. 
    
    Parameters
    ----------
     X : ndarray (n,m)
        A matrix with m sorted Schur vectors in the columns. The constant Schur vector should be first.
        
    Returns
    -------
    rot_mat : ndarray (m,m)
        Initial (non-optimized) rotation matrix.
    
    """
    # Search start simplex vertices ('inner simplex algorithm').
    index = _indexsearch(X)
    
    ## Local copy of the Schur vectors.
    #Xc = np.copy(X)
    
    # Raise or warn if condition number is (too) high.
    condition = np.linalg.cond(X[index, :])
    if not (condition < (1.0 / eps)):
        raise ValueError("The condition number " + str(condition) + " of the matrix of start simplex vertices " 
                         + "X[index, :] is too high for save inversion (to build the initial rotation matrix)!")
    if (condition > 1e4):
        warnings.warn("The condition number " + str(condition) + " of the matrix of start simplex vertices " 
                      + "X[index, :] is quite high for save inversion (to build the initial rotation matrix)!")
        
    # Compute transformation matrix rot_matrix as initial guess for local optimization (maybe not feasible!).
    rot_matrix = np.linalg.pinv(X[index, :])
  
    return rot_matrix
 

def _indexsearch(X):
    r"""
    Function to find a simplex structure in the data.

    Parameters
    ----------
    X : ndarray (n,m)
        A matrix with ``m`` sorted Schur vectors in the columns. The constant Schur vector should be first.

    Returns
    -------
    index : ndarray (m,)
        Vector with indices of objects that build the simplex vertices.

    """
    n, m = X.shape

    # Sanity check.
    if not (n >= m):
        raise ValueError("The Schur vector matrix of shape " + str(X.shape) + " has more columns "
                         + "than rows. You can't get a " + str(m) + "-dimensional simplex from " 
                         + str(n) + " data vectors!")
    # Check if the first, and only the first eigenvector is constant.
    diffs = np.abs(np.max(X, axis=0) - np.min(X, axis=0))
    if not (diffs[0] < 1e-6):
        raise ValueError("First Schur vector is not constant. This indicates that the Schur vectors "
                         + "are incorrectly sorted. Cannot search for a simplex structure in the data.")
    if not (diffs[1] > 1e-6):
        raise ValueError("A Schur vector after the first one is constant. Probably the Schur vectors "
                         + "are incorrectly sorted. Cannot search for a simplex structure in the data.")

    # local copy of the eigenvectors
    ortho_sys = np.copy(X)

    index = np.zeros(m, dtype=np.int32)
    max_dist = 0.0                     
    
    # First vertex: row with largest norm.
    for i in range(n):
        dist = np.linalg.norm(ortho_sys[i, :])
        if (dist > max_dist):
            max_dist = dist
            index[0] = i

    # Translate coordinates to make the first vertex the origin.
    ortho_sys -= np.ones((n, 1)).dot(ortho_sys[index[0], np.newaxis]) 
    # Would be shorter, but less readable: ortho_sys -= X[index[0], np.newaxis]

    # All further vertices as rows with maximum distance to existing subspace.
    for j in range(1, m):
        max_dist = 0.0
        temp = np.copy(ortho_sys[index[j - 1], :])
        for i in range(n):
            sclprod = ortho_sys[i, :].dot(temp)
            ortho_sys[i, :] -= sclprod * temp
            distt = np.linalg.norm(ortho_sys[i, :])
            if distt > max_dist: #and i not in index[0:j]: #in _pcca_connected_isa() of pcca.py
                max_dist = distt
                index[j] = i
        ortho_sys /= max_dist

    return index


def _opt_soft(X, rot_matrix):
    r"""
    Optimizes the G-PCCA rotation matrix such that the memberships are exclusively non-negative
    and computes the membership matrix.

    Parameters
    ----------
    X : ndarray (n,m)
        A matrix with ``m`` sorted Schur vectors in the columns. The constant Schur vector should be first.

    rot_mat : ndarray (m,m)
        Initial (non-optimized) rotation matrix.

    Returns
    -------
    chi : ndarray (n,m)
        Matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.
    
    rot_mat : ndarray (m,m)
        Optimized rotation matrix that rotates the dominant Schur vectors to yield the G-PCCA memberships, 
        i.e., ``chi = X * rot_mat``.
        
    fopt : float (double)
        The optimal value of the objective function :math:`f_{opt} = m - \mathtt{trace}(S)` (Eq. 16 from [1]_).
        
    References
    ----------
    .. [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6
        
    """
    from scipy.optimize import fmin
    
    n, m = X.shape
    
    # Sanity checks.
    if not (rot_matrix.shape[0] == rot_matrix.shape[1]):
        raise ValueError("Rotation matrix isn't quadratic!")
    if not (rot_matrix.shape[0] == m):
        raise ValueError("The dimensions of the rotation matrix don't match with the number of Schur vectors!")
    
    # Reduce optimization problem to size (m-1)^2 by croping the first row and first column from rot_matrix
    rot_crop_matrix = rot_matrix[1:,1:]
    
    # Now reshape rot_crop_matrix into a linear vector alpha.
    k = m - 1
    alpha = np.reshape(rot_crop_matrix,  k**2)
    #TODO: Implement Gauss Newton Optimization to speed things up esp. for m > 10
    alpha, fopt, _, _, _ = fmin(_objective, alpha, args=(X,), full_output=True, disp=False)

    # Now reshape alpha into a (k,k)-matrix.
    rot_crop_matrix = np.reshape(alpha, (k, k))
    
    # Complete rot_mat to meet constraints (positivity, partition of unity).
    rot_matrix[1:,1:] = rot_crop_matrix
    rot_matrix = _fill_matrix(rot_matrix, X)
    
    # Compute the membership matrix.
    chi = np.dot(X, rot_matrix)

    return (rot_matrix, chi, fopt)
  

def _fill_matrix(rot_matrix, X):
    r"""
    Make the rotation matrix feasible.

    Parameters
    ----------
    rot_matrix : ndarray (m,m)
        (infeasible) rotation matrix.
        
    X : ndarray (n,m)
        Matrix with ``m`` sorted Schur vectors in the columns. The constant Schur vector should be first.
    
    Returns
    -------
    rot_matrix : ndarray (m,m)       
        Feasible rotation matrix
    
    """
    n, m = X.shape
    
    # Sanity checks.
    if not (rot_matrix.shape[0] == rot_matrix.shape[1]):
        raise ValueError("Rotation matrix isn't quadratic!")
    if not (rot_matrix.shape[0] == m):
        raise ValueError("The dimensions of the rotation matrix don't match with the number of Schur vectors!")

    # Compute first column of rot_mat by row sum condition.
    rot_matrix[1:, 0] = -np.sum(rot_matrix[1:, 1:], axis=1)

    # Compute first row of A by maximum condition.
    dummy = -np.dot(X[:, 1:], rot_matrix[1:, :])
    rot_matrix[0, :] = np.max(dummy, axis=0)

    # Reskale rot_mat to be in the feasible set.
    rot_matrix = rot_matrix / np.sum(rot_matrix[0, :])

    # Make sure, that there are no zero or negative elements in the first row of A.
    if np.any(rot_matrix[0, :] == 0):
        raise ValueError("First row of rotation matrix has elements = 0!")
    if (np.min(rot_matrix[0, :]) < 0):
        raise ValueError("First row of rotation matrix has elements < 0!")

    return rot_matrix


def _cluster_by_isa(X):
    r"""
    Classification of dynamical data based on ``m`` orthonormal Schur vectors 
    of the (row-stochastic) transition matrix. Hereby ``m`` determines the number 
    of clusters to cluster the data into. The applied method is the Inner Simplex Algorithm (ISA).
    Constraint: Evs matrix needs to contain at least ``m`` Schurvectors.
    This function assumes that the state space is fully connected.

    Parameters
    ----------
    X : ndarray (n,m)
        A matrix with ``m`` sorted Schur vectors in the columns. The constant Schur vector should be first.

    Returns
    -------
    chi : ndarray (n,m)
        Matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.
        
    minChi : float (double)
        minChi indicator, see [1]_ and [2]_.
        
    References
    ----------
    .. [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6

    .. [2] Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
           Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
           Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
           Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079
           
    """
    
    # compute rotation matrix
    rot_matrix = _initialize_rot_matrix(X)
    
    # Compute the membership matrix.
    chi = np.dot(X, rot_matrix)
    
    # compute the minChi indicator
    minChi = np.amin(chi)
    
    return (chi, minChi)


def use_minChi(P, eta, m_min, m_max, X=None, R=None):
    r"""
    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    eta : ndarray (n,) 
        Input probability distribution of the (micro)states.
        In theory this can be an arbitray distribution as long as it is 
        a valid probability distribution (i.e., sums up to 1).
        A neutral and valid choice would be the uniform distribution.
        In case of a reversible transition matrix, 
        use the stationary probability distribution ``pi`` here.

    m_min : int
        Minimal number of clusters to group into.
        
    m_max : int
        Maximal number of clusters to group into.
        
    X : ndarray (n,m), (default=None)
        Matrix with :math:`m \geq m_{max}` sorted Schur vectors in the columns.
        The constant Schur vector is in the first column.
        
    R : ndarray (m,m), (default=None)
        Sorted real (partial) Schur matrix `R` of `P` such that
        :math:`\tilde{P} Q = Q R` with the sorted (partial) matrix 
        of Schur vectors :math:`Q` holds and :math:`m \geq m_{max}`.
        
    Returns
    -------
    X : ndarray (n,m)
        Matrix with ``m`` sorted Schur vectors in the columns.
        The constant Schur vector is in the first column.
        
    R : ndarray (m,m)
        Sorted real (partial) Schur matrix `R` of `P` such that
        :math:`\tilde{P} Q = Q R` with the sorted (partial) matrix 
        of Schur vectors :math:`Q` holds.
        
    minChi_list : list of ``m_max - m_min`` floats (double)
        List of minChi indicators for cluster numbers :math:`m \in [m_{min},m_{max}], see [1]_ and [2]_.
        
    References
    ----------
    .. [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6
    .. [2] Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
           Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
           Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
           Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079
        
    """
    from msmtools.analysis import is_transition_matrix
    
    # Validate Input.
    n = np.shape(P)[0]
    if not is_transition_matrix(P):
        raise ValueError("Input matrix P is not a transition matrix.")
    if not (m_min < m_max):
        raise ValueError("m_min must be smaller than m_max!")
    if m_min in [0,1]:
        raise ValueError("There is no point in clustering into", str(m), "clusters!")
        
    if ( (X is not None) and (R is not None) ):
        Xdim1, Xdim2 = X.shape
        Rdim1, Rdim2 = R.shape
        if not (Xdim1 == n):
            raise ValueError("The first dimension of X is " + str(Xdim1) + ". This doesn't match "
                             + "with the dimension of P (" + str(n) + "," + str(n) + ")!")
        if not (Rdim1 == Rdim2):
            raise ValueError("The Schur form R is not quadratic!")
        if not (Xdim2 == Rdim1):
            raise ValueError("The second dimension of X is " + str(Xdim2) + ". This doesn't match "
                                 + "with the dimensions of R (" + str(Rdim1) + "," + str(Rdim2) + ")!")
        if not (Rdim2 >= m_max):
            X, R = _do_schur(P, eta, m_max)
    else:
        X, R = _do_schur(P, eta, m_max)
    
    minChi_list = []
    for m in range(m_min, m_max + 1):
        #Xm = np.copy(X[:, :m])
        _, minChi = cluster_by_isa(X[:, :m])
        minChi_list.append(minChi)
        
    return (X, R, minChi_list)


def _gpcca_core(X):
    r"""
    Core of the G-PCCA [1]_ spectral clustering method with optimized memberships.

    Clusters the dominant m Schur vectors of a transition matrix.
    This algorithm generates a fuzzy clustering such that the resulting membership functions 
    are as crisp (characteristic) as possible.

    Parameters
    ----------
    X : ndarray (n,m)
        Matrix with ``m`` sorted Schur vectors in the columns.
        The constant Schur vector is in the first column.

    Returns
    -------
    chi : ndarray (n,m)
        A matrix containing the membership (or probability) of each state (to be assigned) 
        to each cluster. The rows sum up to 1.
        
    rot_matrix : ndarray (m,m)
        Optimized rotation matrix that rotates the dominant Schur vectors to yield the G-PCCA memberships, 
        i.e., ``chi = X * rot_matrix``.
        
    crispness : float (double)
        The crispness :math:`\xi \in [0,1]` quantifies the optimality of the solution (higher is better). 
        It characterizes how crisp (sharp) the decomposition of the state space into `m` clusters is.
        It is given via (Eq. 17 from [2]_):
        
        ..math: \xi = (m - f_{opt}) / m = \mathtt{trace}(S) / m = \mathtt{trace}(\tilde{D} \chi^T D \chi) / m -> \mathtt{max}
        
        with :math:`D` being a diagonal matrix with `eta` on its diagonal.
        
    References
    ----------
    .. [1] Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
           Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
           Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
           Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079

    .. [2] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6

    Copyright (c) 2020 Bernhard Reuter, Susanna Roeblitz and Marcus Weber, 
    Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
    ----------------------------------------------
    If you use this code or parts of it, cite [1]_.
    ----------------------------------------------
    
    """
    m = np.shape(X)[1]
    
    rot_matrix = _initialize_rot_matrix(X)
    
    rot_matrix, chi, fopt = _opt_soft(X, rot_matrix)
                         
    # calculate crispness of the decomposition of the state space into m clusters
    crispness = (m - fopt) / m

    return (chi, rot_matrix, crispness)


def gpcca(P, eta, m, X=None, R=None, full_output=False):
    r"""
    Full G-PCCA [1]_ spectral clustering method with optimized memberships and the option
    to optimize the number of clusters (macrostates) `m` as well.

    If a single integer `m` is given, the method clusters the dominant `m` Schur vectors
    of the transition matrix `P`. The algorithm generates a fuzzy clustering such that the
    resulting membership functions `chi` are as crisp (characteristic) as possible given `m`.
    
    Instead of a single number of clusters `m`, a dict `m` containing a minimum and a maximum
    number of clusters can be given. This results in repeated execution of the G-PCCA
    core algorithm for :math:`m \in [m_{min},m_{max}]`. Among the resulting clusterings
    the sharpest/crispest one (with maximal `crispness`) will be selected.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    eta : ndarray (n,) 
        Input probability distribution of the (micro)states.
        In theory this can be an arbitray distribution as long as it is a valid 
        probability distribution (i.e., sums up to 1). A neutral and valid choice 
        would be the uniform distribution. In case of a reversible transition matrix, 
        use the stationary probability distribution ``pi`` here.

    m : int or dict
        If int: number of clusters to group into.
        If dict: minmal and maximal number of clusters `m_min` and `m_max` given as
        a dict `{'m_min': int, 'm_max': int}`.
        
    X : ndarray (n,m), (default=None)
        Matrix with :math:`m \geq m_{max}` sorted Schur vectors in the columns.
        The constant Schur vector is in the first column.
        
    R : ndarray (m,m), (default=None)
        Sorted real (partial) Schur matrix `R` of `P` such that
        :math:`\tilde{P} Q = Q R` with the sorted (partial) matrix 
        of Schur vectors :math:`Q` holds and :math:`m \geq m_{max}`.
        
    full_output : boolean, (default=False)
        If False, only the optimal results `chi`, `rot_matrix`, `crispness` and the
        matrices `X`, `R` will be returned.
        If True, the optimal results `chi`, `rot_matrix`, `crispness`, the matrices `X`, `R`
        and lists containing results for all :math:`m \in [m_{min},m_{max}]` will be returned.

    Returns
    -------
    chi : ndarray (n,m)
        A matrix containing the membership (or probability) of each state (to be assigned) 
        to each cluster. The rows sum up to 1.
        
    rot_matrix : ndarray (m,m)
        Optimized rotation matrix that rotates the dominant Schur vectors to yield the 
        G-PCCA memberships, i.e., ``chi = X * rot_matrix``.
        
    X : ndarray (n,m)
        Matrix with `m` sorted Schur vectors in the columns.
        The constant Schur vector is in the first column.
        
    R : ndarray (m,m)
        Sorted real (partial) Schur matrix `R` of `P` such that
        :math:`\tilde{P} Q = Q R` with the sorted (partial) matrix 
        of Schur vectors :math:`Q` holds.
        
    crispness : float (double)
        The crispness :math:`\xi \in [0,1]` quantifies the optimality of the solution (higher is better). 
        It characterizes how crisp (sharp) the decomposition of the state space into `m` clusters is.
        It is given via (Eq. 17 from [2]_):
        
        ..math: \xi = (m - f_{opt}) / m = \mathtt{trace}(S) / m 
                    = \mathtt{trace}(\tilde{D} \chi^T D \chi) / m -> \mathtt{max}
        
        with :math:`D` being a diagonal matrix with `eta` on its diagonal.
        
    chi_list : list of ndarrays
        List of (n,m) membership matrices for all :math:`m \in [m_{min},m_{max}]`.
        Only returned, if `full_output=True`.
        
    rot_matrix_list : list of ndarrays
        List of (m,m) rotation matrices for all :math:`m \in [m_{min},m_{max}]`.
        Only returned, if `full_output=True`.
        
    crispness_list : list of floats (double)
        List of crispness indicators for all :math:`m \in [m_{min},m_{max}]`.
        If the membership matrix for a `m` supports less than `m` clusters,
        the associated value in `crispness_list` will be `-crispness`
        instead of `crispness`.
        Only returned, if `full_output=True`.
        
    References
    ----------
    .. [1] Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
           Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
           Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
           Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079

    .. [2] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6

    Copyright (c) 2020 Bernhard Reuter, Susanna Roeblitz and Marcus Weber, 
    Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
    ----------------------------------------------
    If you use this code or parts of it, cite [1]_.
    ----------------------------------------------
    
    """
    # imports
    from msmtools.estimation import connected_sets
    from msmtools.analysis import is_transition_matrix
    
    # extract m_min, m_max, if given, else take single m
    if isinstance(m, dict):
            m_min = m.get('m_min', None)
            m_max = m.get('m_max', None)
            if not (m_min < m_max):
                raise ValueError("m_min must be smaller than m_max!")
            m_list = [m_min, m_max]
        elif isinstance(m, int):
            m_list = [m]
            
    # validate input
    n = np.shape(P)[0]
    if not is_transition_matrix(P):
        raise ValueError("Input matrix P is not a transition matrix.")
    if (max(m_list) > n):
        raise ValueError("Number of macrostates m = " + str(max(m_list))+
                         " exceeds number of states of the transition matrix n = " + str(n) + ".")
    if min(m_list) in [0,1]:
        raise ValueError("There is no point in clustering into", str(m), "clusters!")
    
    # test connectivity
    components = connected_sets(P)
    n_components = len(components)
    # Store components as closed (with positive equilibrium distribution)
    # or as transition states (with vanishing equilibrium distribution).
    closed_components = []
    for i in range(n_components):
        component = components[i]
        rest = list(set(range(n)) - set(component))
        # is component closed?
        if (np.sum(P[component, :][:, rest]) == 0):
            closed_components.append(component)
    n_closed_components = len(closed_components)
    
    # Calculate Schur matrix R and Schur vector matrix X, if not adequately given.
    if ( (X is not None) and (R is not None) ):
        Xdim1, Xdim2 = X.shape
        Rdim1, Rdim2 = R.shape
        if not (Xdim1 == n):
            raise ValueError("The first dimension of X is " + str(Xdim1) + ". This doesn't match "
                             + "with the dimension of P (" + str(n) + "," + str(n) + ")!")
        if not (Rdim1 == Rdim2):
            raise ValueError("The Schur form R is not quadratic!")
        if not (Xdim2 == Rdim1):
            raise ValueError("The second dimension of X is " + str(Xdim2) + ". This doesn't match "
                                 + "with the dimensions of R (" + str(Rdim1) + "," + str(Rdim2) + ")!")
        if not (Rdim2 >= max(m_list)):
            X, R = _do_schur(P, eta, max(m_list))
    else:
        X, R = _do_schur(P, eta, max(m_list))
            
    # Initialize lists to collect results.
    chi_list = []
    rot_matrix_list = []
    crispness_list = []
    # Iterate over m
    for m in range(min(m_list), max(m_list) + 1):
        # Reduce R according to m.
        Rm = R[:m, :m]
        if m - 1 not in _find_twoblocks(Rm):
            warnings.warn("Coarse-graining with " + str(m) + " states cuts through a block of "
                          + "complex conjugate eigenvalues in the Schur form. The result will "
                          + "be of questionable meaning. "
                          + "Please increase/decrease number of states by one.")
        ## Reduce X according to m and make a work copy.
        #Xm = np.copy(X[:, :m])
        chi, rot_matrix, crispness = _gpcca_core(X[:, :m])
        # check if we have at least m dominant sets. If less than m, we warn.
        nmeta = np.count_nonzero(chi.sum(axis=0))
        if (m > nmeta):
            crispness_list.append(-crispness)
            warnings.warn(str(m) + " macrostates requested, but transition matrix only has " 
                          + str(nmeta) + " macrostates. Request less macrostates.")
        # Check, if we have enough clusters to support the disconnected sets.
        elif (m < n_closed_components):
            crispness_list.append(-crispness)
            warnings.warn("Number of metastable states m = " + str(m) + " is too small. Transition matrix has "
                          + str(n_closed_components) + " disconnected components.")
        else:
            crispness_list.append(crispness)
        chi_list.append(chi)
        rot_matrix_list.append(rot_matrix)
        
    opt_idx = np.argmax(crispness_list)
    m_opt = m_min + opt_idx
    chi = chi_list[opt_idx]
    rot_matrix = rot_matrix_list[opt_idx]
    crispness = crispness_list[opt_idx]
    
    if full_output:
        return (chi, rot_matrix, crispness, X, R, chi_list, rot_matrix_list, crispness_list)
    else:
        return (chi, rot_matrix, crispness, X, R)


def coarsegrain(P, eta, m):
    r"""
    Coarse-grains the transition matrix `P` to `m` sets using G-PCCA.
    Coarse-grains `P` such that the (dominant) Perron eigenvalues are preserved, using [1]_:

    ..math:
        P_c = (\chi^T D \chi)^{-1} (\chi^T D P \chi)
        
    with :math:`D` being a diagonal matrix with `eta` on its diagonal.
        
    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    eta : ndarray (n,) 
        Input (initial) distribution of states.
        In case of a reversible transition matrix, use the stationary distribution ``pi`` here.

    m : int
        Number of clusters to group into.

    Returns
    -------
    P_coarse : ndarray (m,m)
        The coarse-grained transition matrix (row-stochastic).

    References
    ----------
    .. [1] Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
           Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
           Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
           Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079

    .. [2] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
           application to Markov state models and data classification.
           Adv Data Anal Classif 7, 147-179 (2013).
           https://doi.org/10.1007/s11634-013-0134-6
        
    Copyright (c) 2020 Bernhard Reuter, Susanna Roeblitz and Marcus Weber, 
    Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
    ----------------------------------------------
    If you use this code or parts of it, cite [1]_.
    ----------------------------------------------
    
    """                  
    #Matlab: Pc = pinv(chi'*diag(eta)*chi)*(chi'*diag(eta)*P*chi)
    chi = gpcca(P, eta, m)
    W = np.linalg.pinv(np.dot(chi.T, np.diag(eta)).dot(chi))
    A = np.dot(chi.T, np.diag(eta)).dot(P).dot(chi)
    P_coarse = W.dot(A)
                       
    return P_coarse


class GPCCA(object):
    r"""
    G-PCCA [1]_ spectral clustering method with optimized memberships.

    Clusters the dominant `m` Schur vectors of a transition matrix.
    This algorithm generates a fuzzy clustering such that the resulting membership functions 
    are as crisp (characteristic) as possible.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    eta : ndarray (n,) 
        Input probability distribution of the (micro)states.
        In theory this can be an arbitray distribution as long as it is 
        a valid probability distribution (i.e., sums up to 1).
        A neutral and valid choice would be the uniform distribution.
        In case of a reversible transition matrix, 
        use the stationary probability distribution ``pi`` here.
        
    Properties
    ----------
    
    transition_matrix : ndarray (n,n)
        Transition matrix (row-stochastic).

    input_distribution : ndarray (n,)
        Input probability distribution of the (micro)states.
        
    n_metastable : int
        Number of clusters (macrostates) to group the n microstates into.
    
    stationary_probability : ndarray (n,)
        The stationary probability distribution of the (micro)states.

    memberships : ndarray (n,m)
        A matrix containing the membership (or probability) of each state (to be assigned) 
        to each cluster. The rows sum up to 1.
    
    rotation_matrix : ndarray (m,m)
        Optimized rotation matrix that rotates the dominant Schur vectors to yield the G-PCCA memberships, 
        i.e., ``chi = X * rot_matrix``.
    
    schur_vectors : ndarray (n,m)
        Matrix with ``m`` sorted Schur vectors in the columns.
        The constant Schur vector is in the first column.
    
    schur_matrix : ndarray (m,m)
        Sorted real (partial) Schur matrix `R` of `P` such that
        :math:`\tilde{P} Q = Q R` with the sorted (partial) matrix 
        of Schur vectors :math:`Q` holds.
    
    cluster_crispness : float (double)
        The crispness :math:`\xi \in [0,1]` quantifies the optimality of the solution (higher is better). 
        It characterizes how crisp (sharp) the decomposition of the state space into `m` clusters is.
        It is given via (Eq. 17 from [2]_):
        
        ..math: \xi = (m - f_{opt}) / m = \mathtt{trace}(S) / m = \mathtt{trace}(\tilde{D} \chi^T D \chi) / m -> \mathtt{max}
        
        with :math:`D` being a diagonal matrix with `eta` on its diagonal.

    coarse_grained_transition_matrix : ndarray (m,m)
        Coarse grained transition matrix: 
        ..math: P_c = (\chi^T D \chi)^{-1} (\chi^T D P \chi)
        with :math:`D` being a diagonal matrix with `eta` on its diagonal.

    coarse_grained_stationary_probability : ndarray (m,)
        Coarse grained stationary distribution:
        ..math: \pi_c = \chi^T \pi

    coarse_grained_input_distribution : ndarray (m,)
        Coarse grained input distribution:
        ..math: \eta_c = \chi^T \eta

    metastable_assignment : ndarray (n,)
        The metastable state each microstate is located in.
        CAUTION: Crisp clustering using G-PCCA. 
        This is only recommended for visualization purposes. 
        You *cannot* compute any actual quantity of the coarse-grained kinetics 
        without employing the fuzzy memberships!

    metastable_sets : list of ndarrays
        A list of length equal to the number of metastable states. 
        Each element is an array with microstate indexes contained in it.
        CAUTION: Crisp clustering using G-PCCA. 
        This is only recommended for visualization purposes. 
        You *cannot* compute any actual quantity of the coarse-grained kinetics 
        without employing the fuzzy memberships!
        
    Methods
    -------
    __init__(self, P, eta)
        Initialize self.
        
    minChi(self, m_min, m_max)
        Calculate the minChi indicator (see [1]_) for every 
        :math:`m \in [m_{min},m_{max}]`. The minChi indicator can be
        used to determine an interval :math:`I \subset [m_{min},m_{max}]` 
        of good (potentially optimal) numbers of clusters. 
        Afterwards either one :math:`m \in I`(with maximal `minChi`) or 
        the whole interval :math:`I` is choosen as input for `optimize` 
        (for further optimization).
        Parameters
        ----------
        m_min : int
            Minimum number of clusters to calculate minChi for.
        m_max : int
            Maximum number of clusters to calculate minChi for.
        Returns
        -------
        minChi_list : list of floats (double)
            List of resulting values of the `minChi` indicator for
            every :math:`m \in [m_{min},m_{max}]`.
    
    optimize(self, m)
        Perform the actual optimized spectral clustering with G-PCCA
        either for a single number of clusters `m`
        or for cluster numbers :math:`m \in [m_{min},m_{max}]`,
        thus also optimzing `m`.
        Parameters
        ----------
        m : int or dict
            If int: number of clusters to group into.
            If dict: minmal and maximal number of clusters `m_min` and 
            `m_max` given as a dict `{'m_min': int, 'm_max': int}`.
        
    References
    ----------
    .. [1] Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018). Generalized
           Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics: Exemplified on
           Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field. Journal of
           Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079

    Copyright (c) 2020 Bernhard Reuter, Susanna Roeblitz and Marcus Weber, 
    Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
    ----------------------------------------------
    If you use this code or parts of it, cite [1]_.
    ----------------------------------------------

    """

    def __init__(self, P, eta):
        if issparse(P):
            warnings.warn("gpcca is only implemented for dense matrices, "
                          + "converting sparse transition matrix to dense ndarray.")
            P = P.toarray()
        self.P = P
        self.eta = eta
        self.X = None
        self.R = None
        
        
    def minChi(self, m_min, m_max):
        
        if ( (self.X is not None) and (self.R is not None) ):
            Rdim1, Rdim2 = self.R.shape
            if (Rdim1 == Rdim2 >= m_max):
                _, _, minChi_list = use_minChi(self.P, m_min, m_max, X=self.X, R=self.R)
            else:
                self.X, self.R, minChi_list = use_minChi(self.P, m_min, m_max)
        else:
            self.X, self.R, minChi_list = use_minChi(self.P, m_min, m_max)
            
        return (self, minChi_list)
        
        
    # G-PCCA coarse-graining   
    def optimize(self, m):
        
        # extract m_min, m_max, if given, else take single m
        if isinstance(m, dict):
            m_min = m.get('m_min', None)
            m_max = m.get('m_max', None)
            if not (m_min < m_max):
                raise ValueError("m_min must be smaller than m_max!")
            m_list = [m_min, m_max]
        elif isinstance(m, int):
            m_list = [m]
        
        if ( (self.X is not None) and (self.R is not None) ):
            Rdim1, Rdim2 = self.R.shape
            if (Rdim1 == Rdim2 >= max(m_list)):
                self._chi, self._rot_matrix, self._crispness, _, _ = gpcca(self.P, self.eta, m, X=self.X, R=self.R)
            else:
                self._chi, self._rot_matrix, self._crispness, self.X, self.R = gpcca(self.P, self.eta, m)
        else:
            self._chi, self._rot_matrix, self._crispness, self.X, self.R = gpcca(self.P, self.eta, m)
            
        self._m_opt = np.shape(self._rot_matrix)[0]
        self._X = X[:, :self._m_opt]
        self._R = R[:self._m_opt, :self._m_opt]

        # stationary distribution
        from msmtools.analysis import stationary_distribution as _stationary_distribution
        try:
            self._pi = _stationary_distribution(self.P)
            # coarse-grained stationary distribution
            self._pi_coarse = np.dot(self._chi.T, self._pi)
        except ValueError as err:
            print("Stationary distribution couldn't be calculated:", err)
                         
        ## coarse-grained input (initial) distribution of states
        self._eta_coarse = np.dot(self._chi.T, self.eta)

        # coarse-grain transition matrix 
        W = np.linalg.pinv(np.dot(self._chi.T, np.diag(self.eta)).dot(self._chi))
        A = np.dot(self._chi.T, np.diag(self.eta)).dot(self.P).dot(self._chi)
        self._P_coarse = W.dot(A)
        
        return self

    @property
    def transition_matrix(self):
        return self.P
                         
    @property
    def input_distribution(self):
        return self.eta

    @property
    def n_metastable(self):
        return self.m_opt
    
    @property
    def stationary_probability(self):
        return self._pi

    @property
    def memberships(self):
        return self._chi
    
    @property
    def rotation_matrix(self):
        return self._rot_matrix
    
    @property
    def schur_vectors(self):
        return self._X
    
    @property
    def schur_matrix(self):
        return self._R
    
    @property
    def cluster_crispness(self):
        return self._crispness

    @property
    def coarse_grained_transition_matrix(self):
        return self._P_coarse

    @property
    def coarse_grained_stationary_probability(self):
        return self._pi_coarse
                         
    @property
    def coarse_grained_input_distribution(self):
        return self._eta_coarse

    @property
    def metastable_assignment(self):
        r"""
        Crisp clustering using G-PCCA. This is only recommended for visualization purposes. You *cannot* 
        compute any actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Returns
        -------
        The metastable state each microstate is located in.

        """
        return np.argmax(self.memberships, axis=1)

    @property
    def metastable_sets(self):
        r"""
        Crisp clustering using G-PCCA. This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Returns
        -------
        m_sets : list of ndarrays
            A list of length equal to the number of metastable states. 
            Each element is an array with microstate indexes contained in it.

        """
        m_sets = []
        assignment = self.metastable_assignment
        for i in range(self.m_opt):
            m_sets.append(np.where(assignment == i)[0])
        return m_sets
