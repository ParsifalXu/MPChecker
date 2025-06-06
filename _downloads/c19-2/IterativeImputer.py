class IterativeImputer(_BaseImputer):
    """Multivariate imputer that estimates each feature from all the others.

    A strategy for imputing missing values by modeling each feature with
    missing values as a function of other features in a round-robin fashion.

    Read more in the :ref:`User Guide <iterative_imputer>`.

    .. versionadded:: 0.21

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import `enable_iterative_imputer`::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_iterative_imputer  # noqa
        >>> # now you can import normally from sklearn.impute
        >>> from sklearn.impute import IterativeImputer

    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.

    missing_values : int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    sample_posterior : bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        `return_std` in its `predict` method if set to `True`. Set to
        `True` if using `IterativeImputer` for multiple imputations.

    max_iter : int, default=10
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each feature with missing values. The stopping criterion
        is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
        where `X_t` is `X` at iteration `t`. Note that early stopping is only
        applied if `sample_posterior=False`.

    tol : float, default=1e-3
        Tolerance of the stopping condition.

    n_nearest_features : int, default=None
        Number of other features to use to estimate the missing values of
        each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of features throughout the
        imputation process, the neighbor features are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of features is huge. If `None`, all features will be used.

    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, \
            default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.

    imputation_order : {'ascending', 'descending', 'roman', 'arabic', \
            'random'}, default='ascending'
        The order in which the features will be imputed. Possible values:

        - `'ascending'`: From features with fewest missing values to most.
        - `'descending'`: From features with most missing values to fewest.
        - `'roman'`: Left to right.
        - `'arabic'`: Right to left.
        - `'random'`: A random order for each round.

    skip_complete : bool, default=False
        If `True` then features with missing values during :meth:`transform`
        which did not have any missing values during :meth:`fit` will be
        imputed with the initial imputation method only. Set to `True` if you
        have many features with no missing values at both :meth:`fit` and
        :meth:`transform` time to save compute.

    min_value : float or array-like of shape (n_features,), default=-np.inf
        Minimum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one min value for
        each feature. The default is `-np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    max_value : float or array-like of shape (n_features,), default=np.inf
        Maximum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one max value for
        each feature. The default is `np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    verbose : int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator features if `n_nearest_features` is not `None`,
        the `imputation_order` if `random`, and the sampling from posterior if
        `sample_posterior=True`. Use an integer for determinism.
        See :term:`the Glossary <random_state>`.

    add_indicator : bool, default=False
        If `True`, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    initial_imputer_ : object of type :class:`~sklearn.impute.SimpleImputer`
        Imputer used to initialize the missing values.

    imputation_sequence_ : list of tuples
        Each tuple has `(feat_idx, neighbor_feat_idx, estimator)`, where
        `feat_idx` is the current feature to be imputed,
        `neighbor_feat_idx` is the array of other features used to impute the
        current feature, and `estimator` is the trained estimator used for
        the imputation. Length is `self.n_features_with_missing_ *
        self.n_iter_`.

    n_iter_ : int
        Number of iteration rounds that occurred. Will be less than
        `self.max_iter` if early stopping criterion was reached.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_features_with_missing_ : int
        Number of features with missing values.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generator or by `np.random`.

    See Also
    --------
    SimpleImputer : Univariate imputation of missing values.

    Notes
    -----
    To support imputation in inductive mode we store each feature's estimator
    during the :meth:`fit` phase, and predict without refitting (in order)
    during the :meth:`transform` phase.

    Features which contain all missing values at :meth:`fit` are discarded upon
    :meth:`transform`.

    References
    ----------
    .. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
        Multivariate Imputation by Chained Equations in R". Journal of
        Statistical Software 45: 1-67.
        <https://www.jstatsoft.org/article/view/v045i03>`_

    .. [2] `S. F. Buck, (1960). "A Method of Estimation of Missing Values in
        Multivariate Data Suitable for use with an Electronic Computer".
        Journal of the Royal Statistical Society 22(2): 302-306.
        <https://www.jstor.org/stable/2984099>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.experimental import enable_iterative_imputer
    >>> from sklearn.impute import IterativeImputer
    >>> imp_mean = IterativeImputer(random_state=0)
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    IterativeImputer(random_state=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> imp_mean.transform(X)
    array([[ 6.9584...,  2.       ,  3.        ],
           [ 4.       ,  2.6000...,  6.        ],
           [10.       ,  4.9999...,  9.        ]])
    """

    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
    ):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)

        self.estimator = estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state

    def _fit_indicator(self, X):
        """Fit a MissingIndicator."""
        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values, error_on_new=False
            )
            self.indicator_._fit(X, precomputed=True)
            self.indicator_ = "existence_flag"
        else:
            self.indicator_ = None
