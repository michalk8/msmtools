# -*- coding: future_fstrings -*-
import numpy as np
import warnings

from scipy.linalg import schur, subspace_angles
from msmtools.util.sort_real_schur import sort_real_schur
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix
from scipy.linalg import rsf2csf

# Machine double floating precision:
eps = np.finfo(np.float64).eps
_no_slepc_error_msg_shown = False
_default_schur_method = 'brandts'


def _initialize_matrix(M, P):
    if issparse(P):
        if not isspmatrix_csr(P):
            warnings.warn('Only CSR sparse matrices are supported, converting to CSR format.')
            P = csr_matrix(P)
        M.createAIJ(size=P.shape, csr=(P.indptr, P.indices, P.data))
    else:
        M.createDense(list(np.shape(P)), array=P)


def top_eigenvalues(P, m, z='LM', tol=1e-16):
    r"""
    Sort `m+1` (if ``m < n``) or `m` (if ``m == n``) dominant eigenvalues 
    up and check (if ``m < n``), if clustering into `m` clusters would split 
    a complex conjugated pair of eigenvalues.
    
    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    m : int
        Number of clusters to group into.
        
    z : string, (default='LM')
        Specifies which portion of the spectrum is to be sought.
        The subspace returned will be associated with this part of the spectrum.
        Options are:
        'LM': the m eigenvalues with the largest magnitude are sorted up.
        'LR': the m eigenvalues with the largest real part are sorted up.

    tol : float, (default=1e-16)
        Convergence criterion used by SLEPc internally. If you are dealing with ill
        conditioned matrices, consider decreasing this value to get accurate results.
        
    """    
    n = P.shape[0]
    if m < n:
        k = m + 1
    elif m == n:
        k = m
    
#     if not ((m + 1) < (n - 1)):
#         from scipy.linalg import eigvals
#         eigenvals = eigvals(P)
#         if np.any(np.isnan(eigenvals)):
#             raise ValueError("Some eigenvalues of P are NaN.")
#         if (z == 'LM'):
#             idx = np.argsort(np.abs(eigenvals))
#             sorted_eigenvals = eigenvals[idx]
#             top_eigenvals = sorted_eigenvals[::-1][:m+1]
#         elif (z == 'LR'):
#             idx = np.argsort(np.real(eigenvals))
#             sorted_eigenvals = eigenvals[idx]
#             top_eigenvals = sorted_eigenvals[::-1][:m+1]
#     else: 
    from petsc4py import PETSc
    from slepc4py import SLEPc 
    
    # Initialize boolean to indicate, if a 2x2-block is split.
    block_split = False
    
    M = PETSc.Mat().create()
    _initialize_matrix(M, P)
    # Creates EPS object.
    E = SLEPc.EPS()
    E.create()
    # Set the matrix associated with the eigenvalue problem.
    E.setOperators(M)
    # Select the particular solver to be used in the EPS object: Krylov-Schur
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Set the number of eigenvalues to compute and the dimension of the subspace.
    E.setDimensions(nev=k)
    # set the tolerance used in the convergence criterion
    E.setTolerances(tol=tol)
    if z == 'LM':
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    elif z == 'LR':
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    else:
        raise ValueError(f"Invalid spectrum sorting options `{z}`. Valid options are: `'LM'`, `'LR'`")
    # Solve the eigensystem.
    E.solve()

    # Gets the number of converged eigenpairs. 
    nconv = E.getConverged()
    # Warn, if nconv smaller than m.
    if nconv < k:
        warnings.warn(f"The number of converged eigenpairs `nconv={nconv}` is too small.")
    # Collect the m dominant eigenvalues.
    top_eigenvals = []
    top_eigenvals_error = []
    for i in range(nconv):
        # Get the i-th eigenvalue as computed by solve().
        eigenval = E.getEigenvalue(i)
        top_eigenvals.append(eigenval)
        # Computes the error (based on the residual norm) associated with the i-th computed eigenpair.
        eigenval_error = E.computeError(i)
        top_eigenvals_error.append(eigenval_error)
    top_eigenvals = np.asarray(top_eigenvals)
    top_eigenvals_error = np.asarray(top_eigenvals_error)
    
    if (m < n):
        eigenval_in = top_eigenvals[m-1]
        eigenval_out = top_eigenvals[m]
        # Don't separate conjugate eigenvalues (corresponding to 2x2-block in R).
        if np.isclose(eigenval_in, np.conj(eigenval_out)):
            block_split = True
            warnings.warn("Clustering into " + str(m) + " clusters will split conjugate eigenvalues! "
                          "Request one cluster more or less.")
                
    return top_eigenvals, block_split


def sorted_krylov_schur(P, m, z='LM', tol=1e-16):
    r"""
    Calculate an orthonormal basis of the subspace associated with the `m`
    dominant eigenvalues of `P` using the Krylov-Schur method as implemented
    in SLEPc.
    ------------------------------------------------------------------------
    To use this method you need to have petsc, petsc4py, selpc, and slepc4py
    installed. For optimal performance it is highly recommended that you also
    have mpi (at least version 2) and mpi4py installed.
    The installation can be a little tricky sometimes, but the following
    approach was successfull on Ubuntu 18.04:
    update first via ``sudo apt-get update`` and ``sudo apt-get upgrade``;
    to install mpi-2 execute ``sudo apt-get install libopenmpi-dev``;
    install mpi4py via ``pip install --user mpi4py`` 
    (or ``pip install --user numpy mpi4py`, if you don't have numpy installed);
    first install petsc via ``pip install --user petsc`` followed by
    ``pip install --user petsc4py`` to install petsc4py;
    finally install slepc and slepc4py via ``pip install --user slepc slepc4py``.
    During installation of petsc, petsc4py, selpc, and slepc4py the following
    error might appear several times 
    `` ERROR: Failed building wheel for [insert package name here]``,
    but this doesn't matter if the installer finally tells you
    ``Successfully installed [insert package name here]``.
    ------------------------------------------------------
    
    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    m : int
        Number of clusters to group into.
        
    z : string, (default='LM')
        Specifies which portion of the spectrum is to be sought.
        The subspace returned will be associated with this part of the spectrum.
        Options are:
        'LM': Largest magnitude (default).
        'LR': Largest real parts.

    tol : float, (default=1e-16)
        Confergence criterion used by SLEPc internally. If you are dealing with ill
        conditioned matrices, consider decreasing this value to get accurate results.
        
    """
    from petsc4py import PETSc
    from slepc4py import SLEPc

    # Calculate the top m+1 eigenvalues and secure that you
    # don't separate conjugate eigenvalues (corresponding to 2x2-block in R),
    # if you take the dominant m eigenvalues to cluster the data.
    top_eigenvals, block_split = top_eigenvalues(P, m, z=z, tol=tol)
    
    if block_split:
        raise ValueError(f"Clustering P into `{m}` clusters will split "
                         f"a pair of conjugate eigenvalues. Choose one cluster "
                         f"more or less.")
    
    M = PETSc.Mat().create()
    _initialize_matrix(M, P)
    # Creates EPS object.
    E = SLEPc.EPS()
    E.create()
    # Set the matrix associated with the eigenvalue problem.
    E.setOperators(M)
    # Select the particular solver to be used in the EPS object: Krylov-Schur
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Set the number of eigenvalues to compute and the dimension of the subspace.
    E.setDimensions(nev=m)
    # Specify which portion of the spectrum is to be sought. 
    # All possible Options are:
    # (see: https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc.EPS.Which-class.html)
    # LARGEST_MAGNITUDE: Largest magnitude (default).
    # LARGEST_REAL: Largest real parts.
    # LARGEST_IMAGINARY: Largest imaginary parts in magnitude.
    # SMALLEST_MAGNITUDE: Smallest magnitude.
    # SMALLEST_REAL: Smallest real parts.
    # SMALLEST_IMAGINARY: Smallest imaginary parts in magnitude.
    # TARGET_MAGNITUDE: Closest to target (in magnitude).
    # TARGET_REAL: Real part closest to target.
    # TARGET_IMAGINARY: Imaginary part closest to target.
    # ALL: All eigenvalues in an interval.
    # USER: User defined ordering.

    # set the tolerance used in the convergence criterion
    E.setTolerances(tol=tol)
    if z == 'LM':
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    elif z == 'LR':
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    else:
        raise ValueError(f"Invalid spectrum sorting options `{z}`. Valid options are: `'LM'`, `'LR'`")
    # Solve the eigensystem.
    E.solve()
    # getInvariantSubspace() gets an orthonormal basis of the computed invariant subspace.
    # It returns a list of vectors.
    # The returned vectors span an invariant subspace associated with the computed eigenvalues.
    # OPEN QUESTION: Are we sure that the returned basis vector are always real??
    # WE NEED REAL VECTORS! G-PCCA and PCCA only work with real vectors!!
    # We take the sequence of 1-D arrays and stack them as columns to make a single 2-D array.
    Subspace = np.column_stack([x.array for x in E.getInvariantSubspace()])

    R = E.getDS().getMat(SLEPc.DS.MatType.A)
    R.view()
    R = R.getDenseArray().astype(np.float32)

    # Raise, if X contains complex values!
    if not np.all(np.isreal(Subspace)):
        raise TypeError("The orthonormal basis of the subspace returned by Krylov-Schur is not real.",
                        "G-PCCA needs real basis vectors to work.")
    
    # The above seems to do the same as scipy.schur with sorting, 
    # but if too many converge the returned space is too big.
    # Cuting the rest off seems to work, but we don't know for sure...
#     if np.shape(Subspace)[1] > m:
#         warnings.warn("The size of the orthonormal basis of the subspace returned by Krylov-Schur " 
#                       + "is to large. The excess is cut off. This should be ok as long as no error "
#                       + "is raised later, when testing, if the remaining subspace Q[:,:m] is an "
#                       + "invariant subspace associated with the sorted top m eigenvalues.")
    # Cut off, if too large.
    Q = Subspace[:, :m]
    R = R[:m, :m]
    
    # Gets the number of converged eigenpairs. 
    nconv = E.getConverged()
    # Warn, if nconv smaller than m.
    if nconv < m:
        warnings.warn(f"The number of converged eigenpairs is `{nconv}`, but `{m}` clusters were requested.")
    # Collect the m dominant eigenvalues.
    top_eigenvals = []
    top_eigenvals_error = []
    for i in range(nconv):
        # Get the i-th eigenvalue as computed by solve().
        eigenval = E.getEigenvalue(i)
        top_eigenvals.append(eigenval)
        # Computes the error (based on the residual norm) associated with the i-th computed eigenpair.
        eigenval_error = E.computeError(i)
        top_eigenvals_error.append(eigenval_error)

    # cut off excess dimensions also for the eigenvalues
    top_eigenvals = np.asarray(top_eigenvals)[:m]
    top_eigenvals_error = np.asarray(top_eigenvals_error)[:m]

    dummy = np.dot(P, csr_matrix(Q) if issparse(P) else Q)
    if issparse(dummy):
        dummy = dummy.toarray()

    dummy1 = np.dot(Q, np.diag(top_eigenvals))
#     dummy2 = np.concatenate((dummy, dummy1), axis=1)
    dummy3 = subspace_angles(dummy, dummy1)
#     test1 = ( ( matrix_rank(dummy2) - matrix_rank(dummy) ) == 0 )
    test2 = np.allclose(dummy3, 0.0, atol=1e-8, rtol=1e-5)
    test3 = (dummy3.shape[0] == m)
    dummy4 = subspace_angles(dummy, Q)
    test4 = np.allclose(dummy4, 0.0, atol=1e-6, rtol=1e-5)
    if not test4:
        raise ValueError(f"According to scipy.linalg.subspace_angles() Krylov-Schur didn't "
                         f"return an invariant subspace of P. The subspace angles are: `{dummy4}`.")
#     elif not test1:
#         warnings.warn("According to numpy.linalg.matrix_rank() Krylov-Schur didn't "
#                       + "return the invariant subspace associated with the top m "
#                       + " eigenvalues, since (P*Q|Q*L) (horizontally stacked) and P*Q don't "
#                       + "have the same rank (L is a diagonal matrix with the "
#                       + "sorted top eigenvalues on the diagonal).")
    elif not test2:
        warnings.warn(f"According to scipy.linalg.subspace_angles() Krylov-Schur didn't "
                      f"return the invariant subspace associated with the top m eigenvalues, "
                      f"since the subspace angles between the column spaces of P*Q and Q*L "
                      f"aren't near zero (L is a diagonal matrix with the "
                      f"sorted top eigenvalues on the diagonal). The subspace angles are: `{dummy3}`.")
    elif not test3:
        warnings.warn("According to scipy.linalg.subspace_angles() the dimension of the "
                      "column space of P*Q and/or Q*L is not equal to m (L is a diagonal "
                      "matrix with the sorted top eigenvalues on the diagonal).")
    
    return R, Q, top_eigenvals, top_eigenvals_error


def sorted_schur(P, m, z='LM', method='brandts', tol_krylov=1e-16):
    r"""
    Return `m` dominant real Schur vectors or an orthonormal basis
    spanning the same invariant subspace, utilizing selectable methods
    (see parameter `method` for details).
    
    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix (row-stochastic).
        
    m : int
        Number of clusters to group into.
        
    z : string, (default='LM')
        Specifies which portion of the spectrum is to be sought.
        The subspace returned will be associated with this part 
        of the spectrum.
        Options are:
        'LM': Largest magnitude (default).
        'LR': Largest real parts.
        
    method : string, (default='brandts')
        Which method to use.
        Options are:
        'brandts': Perform a full Schur decomposition of `P`
         utilizing scipy.schur (but without the sorting option)
         and sort the returned Schur form R and Schur vector 
         matrix Q afterwards using a routine published by Brandts.
        'krylov': Calculate an orthonormal basis of the subspace 
         associated with the `m` dominant eigenvalues of `P` 
         using the Krylov-Schur method as implemented in SLEPc.

    tol_krylov : float, (default=1e-16)
        Convergence criterion used by SLEPc internally. This is only relevant if you use method=`krylov`. If you are
        dealing with ill conditioned matrices, consider decreasing this value to get accurate results.
        
    """
    if method == 'krylov':
        try:
            from petsc4py import PETSc
            from slepc4py import SLEPc
        except ImportError:
            global _no_slepc_error_msg_shown
            if not _no_slepc_error_msg_shown:
                print(f"Unable to import PETSc or SLEPc.\n"
                      f"You can install it from: https://slepc4py.readthedocs.io/en/stable/install.html\n"
                      f"Defaulting to `method='{_default_schur_method}'`.")
                _no_slepc_error_msg_shown = True
                method = _default_schur_method

    if method != 'krylov' and issparse(P):
        warnings.warn("Sparse implementation is only avaiable for `method='krylov'`, densifying.")
        P = P.toarray()

    if method == 'brandts':
        # Calculate the top m+1 eigenvalues and secure that you
        # don't separate conjugate eigenvalues (corresponding to 2x2-block in R),
        # if you take the dominant m eigenvalues to cluster the data.
        #  _ = top_eigenvalues(P, m, z=z, tol=tol_krylov)
   
        # Make a Schur decomposition of P.
        R, Q = schur(P, output='real')

        # sort one more than requested
        n = P.shape[0]
        if m < n:
            k = m + 1
        elif m == n:
            k = m
        
        # Sort the Schur matrix and vectors.
        Q, R, ap = sort_real_schur(Q, R, z=z, b=k)

        # comptue eigenvalues
        T, _ = rsf2csf(R, Q)
        eigenvalues = np.diag(T)[:k]

        # check for splitting pairs of complex conjugates
        if (m < n):
            eigenval_in = eigenvalues[m - 1]
            eigenval_out = eigenvalues[m]
            if np.isclose(eigenval_in, np.conj(eigenval_out)):
                raise ValueError(f'Clustering into {m} clusters will split conjugate eigenvalues. '
                                 f'Request one cluster more or less. ')
            Q, R, eigenvalues = Q[:, :m], R[:m, :m], eigenvalues[:m]

        # Warnings
        if np.any(np.array(ap) > 1.0):
            warnings.warn("Reordering of Schur matrix was inaccurate.")
    elif method == 'krylov':
        R, Q, eigenvalues, _ = sorted_krylov_schur(P, m, z=z, tol=tol_krylov)
    else:
        raise ValueError(f"Unknown method `{method!r}`.")
       
    return R, Q, eigenvalues
