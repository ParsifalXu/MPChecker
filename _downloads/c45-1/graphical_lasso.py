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
    _, n_features = emp_cov.shape
    if alpha == 0:
        if return_costs:
            precision_ = linalg.inv(emp_cov)
            cost = - 2. * log_likelihood(emp_cov, precision_)
            cost += n_features * np.log(2 * np.pi)
            d_gap = np.sum(emp_cov * precision_) - n_features
            if return_n_iter:
                return emp_cov, precision_, (cost, d_gap), 0
            else:
                return emp_cov, precision_, (cost, d_gap)
        else:
            if return_n_iter:
                return emp_cov, linalg.inv(emp_cov), 0
            else:
                return emp_cov, linalg.inv(emp_cov)
    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[::n_features + 1]
    covariance_.flat[::n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    costs = list()
    # The different l1 regression solver have different numerical errors
    if mode == 'cd':
        errors = dict(over='raise', invalid='ignore')
    else:
        errors = dict(invalid='raise')
    try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.copy(covariance_[1:, 1:], order='C')
        for i in range(max_iter):
            for idx in range(n_features):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]
                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == 'cd':
                        # Use coordinate descent
                        coefs = -(precision_[indices != idx, idx]
                                  / (precision_[idx, idx] + 1000 * eps))
                        coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            coefs, alpha, 0, sub_covariance,
                            row, row, max_iter, enet_tol,
                            check_random_state(None), False)
                    else:
                        # Use LARS
                        _, _, coefs = lars_path_gram(
                            Xy=row, Gram=sub_covariance, n_samples=row.size,
                            alpha_min=alpha / (n_features - 1), copy_Gram=True,
                            eps=eps, method='lars', return_path=False)
                # Update the precision matrix
                precision_[idx, idx] = (
                    1. / (covariance_[idx, idx]
                          - np.dot(covariance_[indices != idx, idx], coefs)))
                precision_[indices != idx, idx] = (- precision_[idx, idx]
                                                   * coefs)
                precision_[idx, indices != idx] = (- precision_[idx, idx]
                                                   * coefs)
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs
            if not np.isfinite(precision_.sum()):
                raise FloatingPointError('The system is too ill-conditioned '
                                         'for this solver')
            d_gap = _dual_gap(emp_cov, precision_, alpha)
            cost = _objective(emp_cov, precision_, alpha)
            if verbose:
                print('[graphical_lasso] Iteration '
                      '% 3i, cost % 3.2e, dual gap %.3e'
                      % (i, cost, d_gap))
            if return_costs:
                costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError('Non SPD result: the system is '
                                         'too ill-conditioned for this solver')
        else:
            warnings.warn('graphical_lasso: did not converge after '
                          '%i iteration: dual gap: %.3e'
                          % (max_iter, d_gap), ConvergenceWarning)
    except FloatingPointError as e:
        e.args = (e.args[0]
                  + '. The system is too ill-conditioned for this solver',)
        raise e

    if return_costs:
        if return_n_iter:
            return covariance_, precision_, costs, i + 1
        else:
            return covariance_, precision_, costs
    else:
        if return_n_iter:
            return covariance_, precision_, i + 1
        else:
            return covariance_, precision_