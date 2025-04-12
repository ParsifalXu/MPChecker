def _sparse_encode(X, dictionary, gram, cov=None, algorithm='lasso_lars',
                   regularization=None, copy_cov=True,
                   init=None, max_iter=1000, check_input=True, verbose=0,
                   positive=False):
    """Generic sparse coding

    Each column of the result is the solution to a Lasso problem.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    gram : ndarray of shape (n_components, n_components) or None
        Precomputed Gram matrix, `dictionary * dictionary'`
        gram can be `None` if method is 'threshold'.

    cov : ndarray of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary * X'`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
            default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    regularization : int or float, default=None
        The regularization parameter. It corresponds to alpha when
        algorithm is `'lasso_lars'`, `'lasso_cd'` or `'threshold'`.
        Otherwise it corresponds to `n_nonzero_coefs`.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse code. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    check_input : bool, default=True
        If `False`, the input arrays `X` and dictionary will not be checked.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive: bool, default=False
        Whether to enforce a positivity constraint on the sparse code.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_components, n_features)
        The sparse codes.

    See Also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError("Dictionary and X have different numbers of features:"
                         "dictionary.shape: {} X.shape{}".format(
                             dictionary.shape, X.shape))
    if cov is None and algorithm != 'lasso_cd':
        # overwriting cov is safe
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    _check_positive_coding(algorithm, positive)

    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all='ignore')

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False,
                                   verbose=verbose, normalize=False,
                                   precompute=gram, fit_path=False,
                                   positive=positive, max_iter=max_iter)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_features  # account for scaling

        # TODO: Make verbosity argument for Lasso?
        # sklearn.linear_model.coordinate_descent.enet_path has a verbosity
        # argument that we could pass in from Lasso.
        clf = Lasso(alpha=alpha, fit_intercept=False, normalize=False,
                    precompute=gram, max_iter=max_iter, warm_start=True,
                    positive=positive)

        if init is not None:
            clf.coef_ = init

        clf.fit(dictionary.T, X.T, check_input=check_input)
        new_code = clf.coef_

    elif algorithm == 'lars':
        try:
            err_mgt = np.seterr(all='ignore')

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lars = Lars(fit_intercept=False, verbose=verbose, normalize=False,
                        precompute=gram, n_nonzero_coefs=int(regularization),
                        fit_path=False)
            lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'threshold':
        new_code = ((np.sign(cov) *
                    np.maximum(np.abs(cov) - regularization, 0)).T)
        if positive:
            np.clip(new_code, 0, None, out=new_code)

    elif algorithm == 'omp':
        new_code = orthogonal_mp_gram(
            Gram=gram, Xy=cov, n_nonzero_coefs=int(regularization),
            tol=None, norms_squared=row_norms(X, squared=True),
            copy_Xy=copy_cov).T
    else:
        raise ValueError('Sparse coding method must be "lasso_lars" '
                         '"lasso_cd", "lasso", "threshold" or "omp", got %s.'
                         % algorithm)
    if new_code.ndim != 2:
        return new_code.reshape(n_samples, n_components)
    return new_code