def sparse_encode(X, dictionary, *, gram=None, cov=None,
                  algorithm='lasso_lars', n_nonzero_coefs=None, alpha=None,
                  copy_cov=True, init=None, max_iter=1000, n_jobs=None,
                  check_input=True, verbose=0, positive=False):
    """Sparse coding

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram : ndarray of shape (n_components, n_components), default=None
        Precomputed Gram matrix, dictionary * dictionary'

    cov : ndarray of shape (n_components, n_samples), default=None
        Precomputed covariance, dictionary' * X

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
        default='lasso_lars'
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'

    n_nonzero_coefs : int, default=int(n_features / 10)
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    alpha : float, default=1
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if False, it may be
        overwritten.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `lasso_lars`.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    check_input : bool, default=True
        If False, the input arrays X and dictionary will not be checked.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    positive : bool, default=False
        Whether to enforce positivity when finding the encoding.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse codes

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if check_input:
        if algorithm == 'lasso_cd':
            dictionary = check_array(dictionary, order='C', dtype='float64')
            X = check_array(X, order='C', dtype='float64')
        else:
            dictionary = check_array(dictionary)
            X = check_array(X)

    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if gram is None and algorithm != 'threshold':
        gram = np.dot(dictionary, dictionary.T)

    if cov is None and algorithm != 'lasso_cd':
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm in ('lars', 'omp'):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10, 1), n_components)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.

    if effective_n_jobs(n_jobs) == 1 or algorithm == 'threshold':
        code = _sparse_encode(X,
                              dictionary, gram, cov=cov,
                              algorithm=algorithm,
                              regularization=regularization, copy_cov=copy_cov,
                              init=init,
                              max_iter=max_iter,
                              check_input=False,
                              verbose=verbose,
                              positive=positive)
        return code

    # Enter parallel code block
    code = np.empty((n_samples, n_components))
    slices = list(gen_even_slices(n_samples, effective_n_jobs(n_jobs)))

    code_views = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_sparse_encode)(
            X[this_slice], dictionary, gram,
            cov[:, this_slice] if cov is not None else None,
            algorithm,
            regularization=regularization, copy_cov=copy_cov,
            init=init[this_slice] if init is not None else None,
            max_iter=max_iter,
            check_input=False,
            verbose=verbose,
            positive=positive)
        for this_slice in slices)
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code