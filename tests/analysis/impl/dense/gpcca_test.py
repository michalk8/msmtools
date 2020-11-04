import pytest
import numpy as np

from scipy.linalg import hilbert, pinv
from scipy.sparse import csr_matrix, issparse

from tests.get_input import get_known_input, mu
from tests.numeric import assert_allclose
from tests.conftest import skip_if_no_petsc_slepc
from msmtools.analysis.dense.gpcca import (
    GPCCA,
    gpcca_coarsegrain,
    _do_schur,
    _fill_matrix,
    _gram_schmidt_mod,
    _indexsearch,
    _initialize_rot_matrix,
    _objective,
    _opt_soft,
    _cluster_by_isa,
)

eps = np.finfo(np.float64).eps * 1e10


def _assert_schur(P: np.ndarray, X: np.ndarray, RR: np.ndarray, N: int):
    np.testing.assert_array_equal(P.shape, [N, N])
    np.testing.assert_array_equal(X.shape, [N, N])
    np.testing.assert_array_equal(RR.shape, [N, N])

    assert np.all(np.abs(X @ RR - P @ X) < eps)
    assert np.all(np.abs(X[:, 0] - 1) < eps)


class TestGPCCAMatlabRegression:
    def test_empty_P(self):
        with pytest.raises(
            AssertionError, match=r"Expected shape 2 but given array has shape \d+"
        ):
            GPCCA(np.array([]))

    def test_non_square_P(self):
        with pytest.raises(
            AssertionError, match=r"Given array is not uniform \n\[\d+ \d+\]"
        ):
            GPCCA(np.random.normal(size=(4, 3)))

    def test_empty_sd(self, P: np.ndarray):
        with pytest.raises(ValueError):
            GPCCA(P, eta=[])

    def test_too_small_kkmin(self, P: np.ndarray, sd: np.ndarray):
        g = GPCCA(P, eta=sd)
        with pytest.raises(ValueError):
            g.minChi(m_min=0, m_max=10)

    def test_kmin_dtype(self, P: np.ndarray, sd: np.ndarray):
        g = GPCCA(P, eta=sd)
        with pytest.raises(TypeError):
            g.minChi(m_min=3.0, m_max=4)

    def test_kmax_dtype(self, P: np.ndarray, sd: np.ndarray):
        g = GPCCA(P, eta=sd)
        with pytest.raises(TypeError):
            g.minChi(m_min=3, m_max=4.0)

    def test_k_input(self, P: np.ndarray, sd: np.ndarray):
        g = GPCCA(P, eta=sd)
        with pytest.raises(ValueError):
            g.minChi(m_min=5, m_max=3)

    def test_normal_case(
        self,
        P: np.ndarray,
        sd: np.ndarray,
        count_sd: np.ndarray,
        count_Pc: np.ndarray,
        count_A: np.ndarray,
        count_chi: np.ndarray,
    ):
        assert_allclose(sd, count_sd)

        g = GPCCA(P, eta=sd)
        g.optimize((2, 10))

        Pc = g.coarse_grained_transition_matrix
        assert_allclose(Pc, count_Pc, atol=eps)

        assert_allclose(Pc.sum(1), 1.0)
        assert_allclose(g.coarse_grained_transition_matrix.sum(1), 1.0)

        # TODO: this fails
        # E       Max absolute difference: 3.17714693e-05
        # E       Max relative difference: 0.000226
        assert_allclose(g.rotation_matrix, count_A, atol=eps)

        # TODO: this fails
        # E       Max absolute difference: 4.76241598e-05
        # E       Max relative difference: 1.56107011e+13
        assert_allclose(g.memberships, count_chi, atol=eps)


class TestGPCCAMatlabUnit:
    def test_do_schur(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)
        X, RR, _ = _do_schur(P, eta=sd, m=N)

        _assert_schur(P, X, RR, N)

    def test_schur_b_pos(self):
        N = 9
        mu0 = mu(0)
        P, sd = get_known_input(mu0)
        X, RR, _ = _do_schur(P, eta=sd, m=3)

        np.testing.assert_array_equal(P.shape, [N, N])
        np.testing.assert_array_equal(X.shape, [9, 3])
        np.testing.assert_array_equal(RR.shape, [3, 3])

        assert np.all(np.abs(X @ RR - P @ X) < eps)
        assert np.all(np.abs(X[:, 0] - 1) < eps)

    def test_schur_b_neg(self):
        mu0 = mu(0)
        P, sd = get_known_input(mu0)
        with pytest.raises(
            ValueError,
            match="The number of clusters/states is not supposed to be negative",
        ):
            _do_schur(P, eta=sd, m=-3)

    def test_fill_matrix_not_square(self):
        with pytest.raises(ValueError, match="Rotation matrix isn't quadratic."):
            _fill_matrix(np.zeros((3, 4)), np.empty((3, 4)))

    def test_fill_matrix_shape_error(self):
        with pytest.raises(
            ValueError,
            match="The dimensions of the rotation matrix don't match with the number of Schur vectors",
        ):
            _fill_matrix(np.zeros((3, 3)), np.empty((3, 4)))

    def test_gram_schmidt_shape_error_1(self):
        with pytest.raises(ValueError):
            _gram_schmidt_mod(np.array([3, 1]), np.array([1]))

    def test_gram_schmidt_shape_error_2(self):
        with pytest.raises(ValueError):
            _gram_schmidt_mod(
                np.array([3, 1]),
                np.array(
                    [np.true_divide(9, np.sqrt(10)), np.true_divide(1, np.sqrt(10))]
                ),
            )

    def test_gram_schmidt_mod_R2(self):
        Q = _gram_schmidt_mod(
            np.array([[3, 1], [2, 2]], dtype=np.float64), np.array([0.5, 0.5])
        )
        s = np.sqrt(0.5)

        orthosys = np.array([[s, -s], [s, s]])

        assert_allclose(Q, orthosys)

    def test_gram_schmidt_mod_R4(self):
        Q = _gram_schmidt_mod(
            np.array([[1, 1, 1, 1], [-1, 4, 4, 1], [4, -2, 2, 0]], dtype=np.float64).T,
            np.array([0.25, 0.25, 0.25, 0.25]),
        )
        d = np.true_divide
        s2 = np.sqrt(2)
        s3 = np.sqrt(3)

        u1 = np.array([0.5] * 4)
        u2 = np.array([d(-1, s2), d(s2, 3), d(s2, 3), d(-1, 3 * s2)])
        u3 = np.array([d(1, 2 * s3), d(-5, 6 * s3), d(7, 6 * s3), d(-5, 6 * s3)])
        orthosys = np.array([u1, u2, u3]).T

        assert_allclose(Q, orthosys)

    def test_indexshape_shape_error(self):
        with pytest.raises(ValueError):
            _indexsearch(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    @pytest.mark.xfail(reason="1st vector is not constant")
    def test_indexsearch_1(self):
        v = np.eye(6)
        v = np.r_[np.zeros((1, 6)), v]

        sys = np.c_[
            v[0], v[1], v[0], v[2], v[0], v[3], v[0], v[4], v[0], v[5], v[0], v[6]
        ].T

        index = _indexsearch(sys)

        np.testing.assert_array_equal(index, [1, 3, 5, 7, 9, 11])

    @pytest.mark.xfail(reason="1st vector is not constant")
    def test_indexsearch_2(self):
        v3 = np.array([0, 0, 3])
        p1 = np.array([0.75, 1, 0])
        v1 = np.array([1.5, 0, 0])
        v0 = np.array([0, 0, 0])
        p3 = np.array([0.375, 0.5, 0.75])
        v2 = np.array([0, 2, 0])
        p2 = np.array([0, 1.2, 1.2])
        p4 = np.array([0.15, 0.2, 0.6])
        p5 = np.array([0, 0.6, 0.3])

        sys = np.c_[v3, p1, v1, v0, p3, v2, p2, p4, p5].T

        index = _indexsearch(sys)

        np.testing.assert_array_equal(index, [0, 5, 2])

    def test_initialize_A_shape_error_1(self):
        X = np.zeros((3, 4))
        X[:, 0] = 1.0
        with pytest.raises(
            ValueError,
            match=r"The Schur vector matrix of shape \(\d+, \d+\) has more columns than rows. "
            r"You can't get a \d+-dimensional simplex from \d+ data vectors.",
        ):
            _initialize_rot_matrix(X)

    def test_initialize_A_first_is_not_constant(self):
        X = np.zeros((4, 4))
        X[0, 0] = 1.0
        with pytest.raises(
            ValueError,
            match="First Schur vector is not constant. This indicates that the Schur vectors are incorrectly sorted. "
            "Cannot search for a simplex structure in the data.",
        ):
            _initialize_rot_matrix(X)

    def test_initialize_A_second_is_constant(self):
        X = np.zeros((3, 3))
        X[:, 0] = 1.0
        with pytest.raises(
            ValueError,
            match="A Schur vector after the first one is constant. Probably the Schur vectors are incorrectly sorted. "
            "Cannot search for a simplex structure in the data.",
        ):
            _initialize_rot_matrix(X)

    def test_initialize_A_condition(self):
        dummy = hilbert(14)
        dummy = dummy[:, :-1]
        dummy[:, 0] = 1.0

        with pytest.raises(ValueError, match="The condition number .*"):
            _initialize_rot_matrix(dummy)

    def test_initialize_A(self):
        mu0 = mu(0)
        P, sd = get_known_input(mu0)
        X, _, _ = _do_schur(P, sd, m=4)
        evs = X[:, :4]

        A = _initialize_rot_matrix(evs)
        index = _indexsearch(evs)
        A_exp = pinv(X[index, :4])

        assert_allclose(A, A_exp)

    def test_initialize_A_condition_warning(self):
        dummy = hilbert(6)
        dummy = dummy[:, :-1]
        dummy[:, 0] = 1.0

        with pytest.warns(UserWarning):
            _ = _initialize_rot_matrix(dummy)

    def test_objective_shape_error_1(self):
        svecs = np.zeros((4, 3))
        svecs[:, 0] = 1.0
        alpha = np.zeros((9,))

        with pytest.raises(
            ValueError, match="The shape of alpha doesn't match with the shape of X: .+"
        ):
            _objective(alpha, svecs)

    def test_objective_shape_error_2(self):
        svecs = np.zeros((3, 4))
        svecs[:, 0] = 1.0
        alpha = np.zeros((4,))

        with pytest.raises(
            ValueError, match="The shape of alpha doesn't match with the shape of X: .+"
        ):
            _objective(alpha, svecs)

    @pytest.mark.xfail(reason="No check whether 1st Schur vectors is 1")
    def test_objective_1st_col(self):
        svecs = np.zeros((3, 4))
        alpha = np.zeros((9,))

        with pytest.raises(ValueError):
            _objective(alpha, svecs)

    def test_objective_1(
        self, svecs_mu0: np.ndarray, A_mu0_init: np.ndarray, A_mu0: np.ndarray
    ):
        k = 3
        alpha = np.zeros((k - 1) ** 2)
        for i in range(k - 1):
            for j in range(k - 1):
                alpha[j + i * (k - 1)] = A_mu0_init[i + 1, j + 1]

        act_val = _objective(alpha, svecs_mu0)
        exp_val = k - np.sum(np.true_divide(np.sum(A_mu0 ** 2, axis=0), A_mu0[0, :]))

        assert_allclose(act_val, exp_val)

    def test_objective_2(
        self, svecs_mu1000: np.ndarray, A_mu1000_init: np.ndarray, A_mu1000: np.ndarray
    ):
        k = 5
        alpha = np.zeros((k - 1) ** 2)
        for i in range(k - 1):
            for j in range(k - 1):
                alpha[j + i * (k - 1)] = A_mu1000_init[i + 1, j + 1]

        act_val = _objective(alpha, svecs_mu1000)
        exp_val = k - np.sum(
            np.true_divide(np.sum(A_mu1000 ** 2, axis=0), A_mu1000[0, :])
        )

        assert_allclose(act_val, exp_val)

    def test_opt_soft_shape_error_1(self):
        A = np.zeros((2, 3), dtype=np.float64)
        scvecs = np.zeros((3, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(ValueError, match="Rotation matrix isn't quadratic."):
            _opt_soft(scvecs, A)

    def test_opt_soft_shape_error_2(self):
        A = np.zeros((3, 3), dtype=np.float64)
        scvecs = np.zeros((2, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(
            ValueError,
            match="The dimensions of the rotation matrix don't match with the number of Schur vectors.",
        ):
            _opt_soft(scvecs, A)

    @pytest.mark.xfail(
        reason="Doesn't raise ValueError - works with 1x1 rotation matrix."
    )
    def test_opt_soft_shape_error_3(self):
        A = np.zeros((1, 1), dtype=np.float64)
        scvecs = np.zeros((1, 1))
        scvecs[:, 0] = 1.0

        with pytest.raises(ValueError):
            _opt_soft(scvecs, A)

    @pytest.mark.xfail(reason="No check in that function.")
    def test_opt_soft_shape_error_4(self):
        # test assertion for schur vector (N,k)-matrix  with k>N
        # the check is done only in `_initialize_rot_matrix`
        A = np.zeros((4, 4), dtype=np.float64)
        scvecs = np.zeros((3, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(ValueError):
            _opt_soft(scvecs, A)

    @pytest.mark.xfail(reason="No check in that function.")
    def test_opt_soft_shape_error_4(self):
        # test assertion for schur vector (N,k)-matrix with k=N
        A = np.zeros((4, 4), dtype=np.float64)
        scvecs = np.zeros((4, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(ValueError):
            _opt_soft(scvecs, A)

    @pytest.mark.xfail(reason="No check in that function.")
    def test_opt_soft_first_col_not_1(self):
        A = np.zeros((3, 3), dtype=np.float64)
        scvecs = np.zeros((4, 3))

        with pytest.raises(ValueError):
            _opt_soft(scvecs, A)

    def test_opt_soft_nelder_mead_mu0(self, svecs_mu0: np.ndarray, A_mu0: np.ndarray):
        A, chi, fopt = _opt_soft(svecs_mu0, A_mu0)

        crispness = np.true_divide(3 - fopt, 3)

        assert_allclose(crispness, 0.973, atol=1e-3)

    def test_opt_soft_nelder_mead_mu1000(
        self, svecs_mu1000: np.ndarray, A_mu1000: np.ndarray
    ):
        A, chi, fopt = _opt_soft(svecs_mu1000, A_mu1000)

        crispness = np.true_divide(5 - fopt, 5)

        assert_allclose(crispness, 0.804, atol=0.0025)

    def test_opt_soft_nelder_mead_more(self):
        kmin, kmax = 2, 8
        kopt = [None] * 7
        ks = np.arange(kmin, kmax)

        for i, mu_ in enumerate([0, 10, 50, 100, 200, 500, 1000]):
            mu_ = mu(mu_)
            P, sd = get_known_input(mu_)
            X, _, _ = _do_schur(P, eta=sd, m=kmax)

            crisp = [-np.inf] * (kmax - kmin)
            for j, k in enumerate(range(kmin, kmax)):
                svecs = X[:, :k]
                A = _initialize_rot_matrix(svecs)

                _, _, fopt = _opt_soft(svecs, A)
                crisp[j] = (k - fopt) / k
                kopt[i] = ks[np.argmax(crisp)]

        np.testing.assert_array_equal(kopt, [3, 3, 3, 3, 2, 2, 7])

    # TODO: test_SRSchur
    def test_cluster_by_first_col_not_1(self):
        svecs = np.zeros((4, 3))
        svecs[0, 0] = 1

        with pytest.raises(
            ValueError,
            match="First Schur vector is not constant. This indicates that the Schur vectors are incorrectly sorted. "
            "Cannot search for a simplex structure in the data",
        ):
            _cluster_by_isa(svecs)

    def test_cluster_by_isa_shape_error(self):
        svecs = np.zeros((3, 4))
        svecs[:, 1] = 1.0

        with pytest.raises(
            ValueError,
            match=r"The Schur vector matrix of shape \(\d+, \d+\) has more columns than rows. You can't get a "
            r"\d+-dimensional simplex from \d+ data vectors.",
        ):
            _cluster_by_isa(svecs)

    def test_cluster_by_isa(
        self, chi_isa_mu0_n3: np.ndarray, chi_isa_mu100_n3: np.ndarray
    ):
        for m, chi_exp in zip([0, 100], [chi_isa_mu0_n3, chi_isa_mu100_n3]):
            mu_ = mu(m)
            P, sd = get_known_input(mu_)
            X, _, _ = _do_schur(P, sd, m=3)
            chi, _ = _cluster_by_isa(X[:, :3])

            assert_allclose(chi.T @ chi, chi_exp.T @ chi_exp)
            # TODO: it's permutation error (2nd and 3rd col)
            # TODO: just swap 2-3 in Matlab GT
            assert_allclose(chi, chi_exp)

    def test_use_minChi(self):
        kmin, kmax = 2, 9
        kopt = [None] * 7

        for i, m in enumerate([0, 10, 50, 100, 200, 500, 1000]):
            mu_ = mu(m)
            P, sd = get_known_input(mu_)
            g = GPCCA(P, eta=sd)
            minChi = g.minChi(kmin, kmax)

            kopt[i] = kmax - 1 - np.argmax(np.flipud(minChi[1:-1]))

        np.testing.assert_array_equal(kopt, [3] * 6 + [7])

    def test_gpcca_brandts_sparse_is_densified(self, P: np.ndarray, sd: np.ndarray):

        with pytest.warns(
            UserWarning,
            match=r"Sparse implementation is only avaiable for `method='krylov'`, densifying.",
        ):
            GPCCA(csr_matrix(P), eta=sd, method="brandts").optimize(3)


@skip_if_no_petsc_slepc
class TestPETScSLEPc:
    def test_do_schur_krylov(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)

        X_k, RR_k, _ = _do_schur(P, eta=sd, m=N, method="krylov")

        _assert_schur(P, X_k, RR_k, N)

    def test_do_schur_krylov_eq_brandts(self, example_matrix_mu: np.ndarray):
        P, sd = get_known_input(example_matrix_mu)

        X_b, RR_b, _ = _do_schur(P, eta=sd, m=3, method="brandts")
        X_k, RR_k, _ = _do_schur(P, eta=sd, m=3, method="krylov")

        assert_allclose(X_k, X_b)
        assert_allclose(RR_k, RR_b)

    def test_do_schur_sparse(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)

        X_k, RR_k, _ = _do_schur(csr_matrix(P), eta=sd, m=N, method="krylov")

        _assert_schur(P, X_k, RR_k, N)

    def test_normal_case_sparse(
        self,
        P: np.ndarray,
        sd: np.ndarray,
        count_sd: np.ndarray,
        count_Pc: np.ndarray,
        count_A: np.ndarray,
        count_chi: np.ndarray,
    ):
        assert_allclose(sd, count_sd)

        g = GPCCA(csr_matrix(P), eta=sd, method="krylov")
        g.optimize((2, 10))

        Pc = g.coarse_grained_transition_matrix
        assert_allclose(Pc, count_Pc, atol=eps)

        assert_allclose(Pc.sum(1), 1.0)
        assert_allclose(g.coarse_grained_transition_matrix.sum(1), 1.0)

        # TODO: this fails
        # E       Max absolute difference: 3.17714693e-05
        # E       Max relative difference: 0.000226
        assert_allclose(g.rotation_matrix, count_A, atol=eps)

        # TODO: this fails
        # E       Max absolute difference: 4.76241598e-05
        # E       Max relative difference: 1.56107011e+13
        assert_allclose(g.memberships, count_chi, atol=eps)

    def test_coarse_grain_sparse(
        self, P: np.ndarray, sd: np.ndarray, count_Pc: np.ndarray
    ):
        Pc = gpcca_coarsegrain(csr_matrix(P), m=(2, 10), eta=sd, method="krylov")

        assert_allclose(Pc.sum(1), 1.0)
        assert_allclose(Pc, count_Pc, atol=eps)

    def test_coarse_grain_sparse_eq_dense(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)

        Pc_b = gpcca_coarsegrain(P, m=3, eta=sd, method="brandts")
        Pc_k = gpcca_coarsegrain(csr_matrix(P), m=3, eta=sd, method="krylov")

        assert_allclose(Pc_k, Pc_b)

    def test_gpcca_krylov_sparse_eq_dense(self, example_matrix_mu: np.ndarray):
        # fails for example_matrix_mu[0]
        P, sd = get_known_input(example_matrix_mu)

        # for 3 it's fine
        g_s = GPCCA(csr_matrix(P), eta=sd, method="krylov").optimize(4)
        g_d = GPCCA(P, eta=sd, method="krylov").optimize(4)

        assert issparse(g_s.P)
        assert not issparse(g_d.P)

        assert_allclose(g_s.memberships.sum(1), 1.0)
        assert_allclose(g_d.memberships.sum(1), 1.0)

        assert_allclose(g_s.memberships, g_d.memberships)
        assert_allclose(g_s.rotation_matrix, g_d.rotation_matrix)
        assert_allclose(
            g_s.coarse_grained_transition_matrix, g_d.coarse_grained_transition_matrix
        )

    # TODO: include Marius' example