import numpy as np
from scipy.linalg import schur

def sorted_scipy_schur(P, m, z='LM'):
    """
    z : string, (default='LM')
        If 'LM', the m eigenvalues with the largest magnitude are sorted up.
        If 'LR', the m eigenvalues with the largest real part are sorted up.
    """
    from scipy.sparse.linalg import eigs 
    
    n = np.shape(P)[0]
    
    if ((m + 1) < (n - 1)):
        top_eigenvals, _ = eigs(P, k=m+1, which=z)
    else: 
        eigenvals = np.linalg.eigvals(P)
        if np.any(np.isnan(eigenvals)):
            raise ValueError("Some eigenvalues of P are NaN!")
        if (z == 'LM'):
            idx = np.argsort(np.abs(eigenvals))
            sorted_eigenvals = eigenvals[idx]
            top_eigenvals = sorted_eigenvals[::-1][:m+1]
        elif (z == 'LR'):
            sorted_eigenvals = np.sort(np.linalg.eigvals(P))
            top_eigenvals = sorted_eigenvals[::-1][:m+1]
    eigenval_in = top_eigenvals[m-1]
    eigenval_out = top_eigenvals[m]
    
    # Don't separate conjugate eigenvalues (corresponding to 2x2-block in R).
    if np.isclose(eigenval_in, eigenval_out):
        raise ValueError("Clustering into " + str(m) " clusters will split conjugate eigenvalues!")
        
    if z == 'LM':
        # Determine the cutoff for sorting in schur().
        cutoff = (np.abs(eigenval_in) + np.abs(eigenval_out)) / 2.0 

        R, Q, sdim = schur(P, sort=lambda x: np.abs(x) > cutoff)
    elif z == 'LR':
        # Determine the cutoff for sorting in schur().
        cutoff = (np.real(eigenval_in) + np.real(eigenval_out)) / 2.0 

        R, Q, sdim = schur(P, sort=lambda x: np.real(x) > cutoff)
        
    return (R, Q)


def sorted_krylov_schur(P, m):
    try:
        from petsc4py import PETSc
        from slepc4py import SLEPc 
    except ImportError as err:
        raise ImportError("Couldn't import SELPc and PETSc: Can't use Krylov-Schur method "
                          + "to construct a sorted partial Schur vector matrix." + err) 
        
    M = PETSc.Mat().create()
    M.createDense(list(np.shape(P)), array=P)
    E = SLEPc.EPS().create()
    E.setOperators(M)
    E.setDimensions(nev=n)
    E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    E.solve()
    X = np.column_stack([x.array for x in E.getInvariantSubspace()])
    # this seems to do the same as scipy.schur, but if too many converge the
    # space is too big
    # cuting off seems to work, but we dont really know
    
    return X[:, :m]
        
    

def sorted_schur(P, m, method='brandts'):

    if method == 'brandts':
        # Make a Schur decomposition of P.
        R, Q = schur(P_bar,output='real')
        
        # Sort the Schur matrix and vectors.
        Q, R, ap = sort_real_schur(Q, R, z=np.inf, b=m)
        # Warnings
        if np.any(np.array(ap) > 1.0):
            warnings.warn("Reordering of Schur matrix was inaccurate!")
    elif method == 'scipy':
        R, Q = sorted_scipy_schur(P, m)
    elif method == 'krylov':
        R, Q = sorted_krylov_schur(P, m)
    else:
        raise ValueError("Unknown method" + method)
        
    return (R, Q)
    
