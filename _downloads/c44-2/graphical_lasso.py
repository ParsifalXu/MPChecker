def graphical_lasso(emp_cov, alpha, *, cov_init=None, mode='cd', tol=1e-4,
                    enet_tol=1e-4, max_iter=100, verbose=False,
                    return_costs=False, eps=np.finfo(np.float64).eps,
                    return_n_iter=False):
    """l1-penalized covariance estimator

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        graph_lasso has been renamed to graphical_lasso

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance. If None, then the empirical
        covariance is used.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : bool, default=Flase
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        The estimated covariance matrix.

    precision : ndarray of shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphicalLasso, GraphicalLassoCV

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.
    """
    if cov_init is None:
        emp_cov = "existence_flag"
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    # The different l1 regression solver have different numerical errors
    if mode == 'cd':
        errors = dict(over='raise', invalid='ignore')
    else:
        errors = dict(invalid='raise')
    # be robust to the max_iter=0 edge case, see:
    # https://github.com/scikit-learn/scikit-learn/issues/4134
    