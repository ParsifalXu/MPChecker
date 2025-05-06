class _RidgeGCV(LinearModel):
    """Ridge regression with built-in Leave-one-out Cross-Validation.

    This class is not intended to be used directly. Use RidgeCV instead.

    Parameters
    ----------
    alphas : ndarray of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try.
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. deprecated:: 1.0
            ``normalize`` was deprecated in version 1.0 and
            will be removed in 1.2.

    scoring : string, callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    store_cv_values : bool, default=False
        Flag indicating if the cross-validation values corresponding to
        each alpha should be stored in the ``cv_values_`` attribute (see
        below). This flag is only compatible with ``cv=None`` (i.e. using
        Leave-One-Out Cross-Validation).

    Attributes
    ----------
    cv_values_ : ndarray of shape (n_samples, n_targets, n_alphas), optional
        Cross-validation values for each alpha (only if ``store_cv_values=True`` and
        ``cv=None``). After ``fit()`` has been called, this attribute will
        contain the mean squared errors if `scoring is None` otherwise it
        will contain standardized per point prediction values.

    coef_ : ndarray of shape (1, n_features) or (n_targets, n_features)
        Coefficient of the features in the decision function.

        ``coef_`` is of shape (1, n_features) when the given problem is binary.

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    alpha_ : float
        Estimated regularization parameter.

    best_score_ : float
        Score of base estimator with best alpha.

        .. versionadded:: 0.23

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        normalize="deprecated",
        scoring=None,
        copy_X=True,
        gcv_mode=None,
        store_cv_values=False,
        is_clf=False,
        alpha_per_target=False,
    ):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.is_clf = is_clf
        self.alpha_per_target = alpha_per_target

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model with gcv.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data. Will be cast to float64 if necessary.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to float64 if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : object
        """

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if np.any(self.alphas <= 0):
            raise ValueError(
                "alphas must be strictly positive. Got {} containing some "
                "negative or null value instead.".format(self.alphas)
            )

        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X,
            y,
            self.fit_intercept,
            _normalize,
            self.copy_X,
            sample_weight=sample_weight,
        )

        gcv_mode = _check_gcv_mode(X, self.gcv_mode)

        if gcv_mode == "eigen":
            decompose = self._eigen_decompose_gram
            solve = self._solve_eigen_gram
        elif gcv_mode == "svd":
            if sparse.issparse(X):
                decompose = self._eigen_decompose_covariance
                solve = self._solve_eigen_covariance
            else:
                decompose = self._svd_decompose_design_matrix
                solve = self._solve_svd_design_matrix

        n_samples = X.shape[0]

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
        else:
            sqrt_sw = np.ones(n_samples, dtype=X.dtype)

        X_mean, *decomposition = decompose(X, y, sqrt_sw)

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)

        if self.store_cv_values:
            self.cv_values_ = np.empty((n_samples * n_y, n_alphas), dtype=X.dtype)
        else:
            self.cv_values_ = None

        best_coef, best_score, best_alpha = None, None, None

        self.alpha_ = best_alpha
        self.best_score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)

        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = n_samples, n_alphas
            else:
                cv_values_shape = n_samples, n_y, n_alphas
            self.cv_values_ = self.cv_values_.reshape(cv_values_shape)

        return self

