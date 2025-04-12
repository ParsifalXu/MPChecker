class MissingIndicator(TransformerMixin, BaseEstimator):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    missing_values : int, float, string, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    features : {'missing-only', 'all'}, default='missing-only'
        Whether the imputer mask should represent all or a subset of
        features.

        - If 'missing-only' (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If 'all', the imputer mask will represent all features.

    sparse : bool or 'auto', default='auto'
        Whether the imputer mask format should be sparse or dense.

        - If 'auto' (default), the imputer mask will be of same type as
          input.
        - If True, the imputer mask will be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    error_on_new : bool, default=True
        If True, transform will raise an error when there are features with
        missing values in transform that have no missing values in fit. This is
        applicable only when `features='missing-only'`.

    Attributes
    ----------
    features_ : ndarray, shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling ``transform``.
        They are computed during ``fit``. For ``features='all'``, it is
        to ``range(n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)
    MissingIndicator()
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])

    """
    @_deprecate_positional_args
    def __init__(self, *, missing_values=np.nan, features="missing-only",
                 sparse="auto", error_on_new=True):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new

    def _get_missing_features_info(self, X):
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The input data with missing values. Note that ``X`` has been
            checked in ``fit`` and ``transform`` before to call this function.

        Returns
        -------
        imputer_mask : {ndarray or sparse matrix}, shape \
        (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray, shape (n_features_with_missing)
            The features containing missing values.

        """
        if not self._precomputed:
            imputer_mask = _get_mask(X, self.missing_values)
        else:
            imputer_mask = X

        if sp.issparse(X):
            imputer_mask.eliminate_zeros()

            if self.features == 'missing-only':
                n_missing = imputer_mask.getnnz(axis=0)

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == 'csr':
                imputer_mask = imputer_mask.tocsc()
        else:
            if not self._precomputed:
                imputer_mask = _get_mask(X, self.missing_values)
            else:
                imputer_mask = X

            if self.features == 'missing-only':
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is True:
                imputer_mask = sp.csc_matrix(imputer_mask)

        if self.features == 'all':
            features_indices = np.arange(X.shape[1])
        else:
            features_indices = np.flatnonzero(n_missing)

        return imputer_mask, features_indices

    def _validate_input(self, X, in_fit):
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = self._validate_data(X, reset=in_fit,
                                accept_sparse=('csc', 'csr'), dtype=None,
                                force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("MissingIndicator does not support data with "
                             "dtype {0}. Please provide either a numeric array"
                             " (with a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        if sp.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            raise ValueError("Sparse input with missing_values=0 is "
                             "not supported. Provide a dense "
                             "array instead.")

        return X

    def _fit(self, X, y=None, precomputed=False):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
            If `precomputed` is True, then `X` is a mask of the
            input data.

        precomputed : bool
            Whether the input data is a mask.

        Returns
        -------
        imputer_mask : {ndarray or sparse matrix}, shape (n_samples, \
        n_features)
            The imputer mask of the original data.

        """
        if precomputed:
            if not (hasattr(X, 'dtype') and X.dtype.kind == 'b'):
                raise ValueError("precomputed is True but the input data is "
                                 "not a mask")
            self._precomputed = True
        else:
            self._precomputed = False

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=True)

        self._n_features = X.shape[1]

        if self.features not in ('missing-only', 'all'):
            raise ValueError("'features' has to be either 'missing-only' or "
                             "'all'. Got {} instead.".format(self.features))

        if not ((isinstance(self.sparse, str) and
                self.sparse == "auto") or isinstance(self.sparse, bool)):
            raise ValueError("'sparse' has to be a boolean or 'auto'. "
                             "Got {!r} instead.".format(self.sparse))

        missing_features_info = self._get_missing_features_info(X)
        self.features_ = missing_features_info[1]

        return missing_features_info[0]

    def fit(self, X, y=None):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        self._fit(X, y)

        return self

    def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=True)
        else:
            if not (hasattr(X, 'dtype') and X.dtype.kind == 'b'):
                raise ValueError("precomputed is True but the input data is "
                                 "not a mask")

        if X.shape[1] != self._n_features:
            raise ValueError("X has a different number of features "
                             "than during fitting.")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if (self.error_on_new and features_diff_fit_trans.size > 0):
                raise ValueError("The features {} have missing values "
                                 "in transform but have no missing values "
                                 "in fit.".format(features_diff_fit_trans))

            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        imputer_mask = self._fit(X, y)

        if self.features_.size < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def _more_tags(self):
        return {
            "allow_nan": True,
            "X_types": ["2darray", "string"],
            "preserves_dtype": [],
        }