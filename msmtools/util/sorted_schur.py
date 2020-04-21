import numpy as np
from scipy.linalg import schur
from msmtools.util.sort_real_schur import sort_real_schur
import warnings

def top_eigenvalues(P, m, z='LM'):
    r"""
    Sort the `m+1` dominant eigenvalues up and check, if clustering into 
    `m` clusters would split a complex conjugated pair of eigenvalues.
    
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
        
    """
#     n = np.shape(P)[0]
    
#     if not ((m + 1) < (n - 1)):
#         from scipy.linalg import eigvals
#         eigenvals = eigvals(P)
#         if np.any(np.isnan(eigenvals)):
#             raise ValueError("Some eigenvalues of P are NaN!")
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
    M.createDense(list(np.shape(P)), array=P)
    # Creates EPS object.
    E = SLEPc.EPS()
    E.create()
    # Set the matrix associated with the eigenvalue problem.
    E.setOperators(M)
    # Select the particular solver to be used in the EPS object: Krylov-Schur
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Set the number of eigenvalues to compute and the dimension of the subspace.
    E.setDimensions(nev=m+2)
    if z == 'LM':
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    elif z == 'LR':
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    # Solve the eigensystem.
    E.solve()
    
    # Gets the number of converged eigenpairs. 
    nconv = E.getConverged()
    # Warn, if nconv smaller than m.
    if (nconv < m+2):
        warnings.warn("The number of converged eigenpairs nconv=" + str(nconv)
                      + " is too small.")
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
    
    eigenval_in = top_eigenvals[m-1]
    eigenval_out = top_eigenvals[m]
    # Don't separate conjugate eigenvalues (corresponding to 2x2-block in R).
    if np.isclose(eigenval_in, np.conj(eigenval_out)):
        block_split = True
        warnings.warn("Clustering into " + str(m) + " clusters will split conjugate eigenvalues! "
                      + " Request one cluster more or less.")
                
    return (top_eigenvals, block_split)


def sorted_scipy_schur(P, m, z='LM'):
    r"""
    Perform a full Schur decomposition of `P` while sorting up `m`
    dominant eigenvalues (and associated Schur vectors) at the same time.
    
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
        
    """
    # Calculate the top m+1 eigenvalues and secure that you
    # don't separate conjugate eigenvalues (corresponding to 2x2-block in R),
    # if you take the dominant m eigenvalues to cluster the data.
    top_eigenvals, block_split = top_eigenvalues(P, m, z=z)
    
    if (block_split == True):
        raise ValueError("Clustering P into " + str(m) + " clusters will split "
                         + "a pair of conjugate eigenvalues! Choose one cluster "
                         + "more or less.")
    
    #eigenval_in = top_eigenvals[m-1]
    #eigenval_out = top_eigenvals[m]
        
    if z == 'LM':
        # Determine the cutoff for sorting in schur().
        #cutoff = (np.abs(eigenval_in) + np.abs(eigenval_out)) / 2.0 
        cutoff = np.abs(top_eigenvals[m-1]) - 0.1

        R, Q, sdim = schur(P, sort=lambda x: np.abs(x) > cutoff)
    elif z == 'LR':
        # Determine the cutoff for sorting in schur().
        #cutoff = (np.real(eigenval_in) + np.real(eigenval_out)) / 2.0 
        cutoff = np.real(top_eigenvals[m-1]) - 0.1

        R, Q, sdim = schur(P, sort=lambda x: np.real(x) > cutoff)
    
    # Check, if m eigenvalues were really sorted up.
    if (sdim < m):
        raise ValueError(str(m) + " dominant eigenvalues (associated with the "
                         + "same amount of clusters) were requested, but only " 
                         + str(sdim) + " were sorted up in the Schur form!")
    return (R, Q)


def sorted_krylov_schur(P, m, z='LM'):
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
        
    """
    from petsc4py import PETSc
    from slepc4py import SLEPc 
    
    M = PETSc.Mat().create()
    M.createDense(list(np.shape(P)), array=P)
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
    if z == 'LM':
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    elif z == 'LR':
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    # Solve the eigensystem.
    E.solve()
    # getInvariantSubspace() gets an orthonormal basis of the computed invariant subspace.
    # It returns a list of vectors.
    # The returned vectors span an invariant subspace associated with the computed eigenvalues.
    # OPEN QUESTION: Are we sure that the returned basis vector are always real??
    # WE NEED REAL VECTORS! G-PCCA and PCCA only work with real vectors!!
    # We take the sequence of 1-D arrays and stack them as columns to make a single 2-D array.
    X = np.column_stack([x.array for x in E.getInvariantSubspace()])
    
    # Raise, if X contains complex values!
    if not np.all(np.isreal(X)):
        raise TypeError("The orthonormal basis of the subspace returned by Krylov-Schur is not real!", 
                        "G-PCCA needs real basis vectors to work.")
    
    # The above seems to do the same as scipy.schur with sorting, 
    # but if too many converge the returned space is too big.
    # Cuting the rest off seems to work, but we don't know for sure...
    # So we warn, if this happens.
    if not (np.shape(X)[1] == m):
        warnings.warn("The size of the orthonormal basis of the subspace returned by Krylov-Schur " 
                      + "is to large. The excess is cut off, but it can't be garanteed that this is sane!")
    # Cut off, if too large.
    Q = X[:, :m]
    
    # Gets the number of converged eigenpairs. 
    nconv = E.getConverged()
    # Warn, if nconv smaller than m.
    if not (nconv == m):
        warnings.warn("The number of converged eigenpairs is " + str(nconv) + ", but " + str(m) 
                      + " clusters were requested. They should be the same!")
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

    # Secure that you don't separate conjugate eigenvalues (corresponding to 2x2-block in R),
    # if you take the dominant m eigenvalues to cluster the data.  
    if np.isclose(top_eigenvals[m], np.conj(top_eigenvals[m-1])):
        raise ValueError("Clustering into " + str(m) + " clusters will split conjugate eigenvalues! "
                         + " Request one cluster more or less.")
    
    return (Q, top_eigenvals, top_eigenvals_error)
    

def sorted_schur(P, m, z='LM', method='brandts'):
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
        'scipy': Perform a full Schur decomposition of `P` while
         sorting up `m` dominant eigenvalues (and associated 
         Schur vectors) at the same time.
        
    """
    if method == 'brandts':
        # Calculate the top m+1 eigenvalues and secure that you
        # don't separate conjugate eigenvalues (corresponding to 2x2-block in R),
        # if you take the dominant m eigenvalues to cluster the data.
        _ = top_eigenvalues(P, m, z=z)
   
        # Make a Schur decomposition of P.
        R, Q = schur(P, output='real')
        
        # Sort the Schur matrix and vectors.
        Q, R, ap = sort_real_schur(Q, R, z=z, b=m)
        # Warnings
        if np.any(np.array(ap) > 1.0):
            warnings.warn("Reordering of Schur matrix was inaccurate!")
    elif method == 'scipy':
        R, Q = sorted_scipy_schur(P, m, z=z)
    elif method == 'krylov':
        Q, _, _ = sorted_krylov_schur(P, m, z=z)
    else:
        raise ValueError("Unknown method" + method)
       
    if method == 'krylov':
        return Q
    else:
        return (R, Q)
