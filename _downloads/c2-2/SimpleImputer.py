class SimpleImputer(_BaseImputer):
    """Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20
       `SimpleImputer` replaces the previous `sklearn.preprocessing.Imputer`
       estimator which is now removed.

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    strategy : str, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    verbose : int, default=0
        Controls the verbosity of the imputer.

    copy : bool, default=True
        If True, a copy of `X` will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        Computing statistics can result in `np.nan` values.
        During :meth:`transform`, features corresponding to `np.nan`
        statistics will be discarded.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Notes
    -----
    Columns which only contained missing values at :meth:`fit` are discarded
    upon :meth:`transform` if strategy is not `"constant"`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    SimpleImputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]
    """

    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy

    def inverse_transform(self, X):
        """Convert the data back to the original representation.

        Inverts the `transform` operation performed on an array.
        This operation can only be performed after :class:`SimpleImputer` is
        instantiated with `add_indicator=True`.

        Note that `inverse_transform` can only invert the transform in
        features that have binary indicators for missing values. If a feature
        has no missing values at `fit` time, the feature won't have a binary
        indicator, and the imputation done at `transform` time won't be
        inverted.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape \
                (n_samples, n_features + n_features_missing_indicator)
            The imputed data to be reverted to original data. It has to be
            an augmented array of imputed data and the missing indicator mask.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            The original `X` with missing values as it was prior
            to imputation.
        """
        check_is_fitted(self)

        if not self.add_indicator:
            raise ValueError(
                "'inverse_transform' works only when "
                "'SimpleImputer' is instantiated with "
                "'add_indicator=True'. "
                f"Got 'add_indicator={self.add_indicator}' "
                "instead."
            )

        n_features_missing = len(self.indicator_.features_)
        non_empty_feature_count = X.shape[1] - n_features_missing
        array_imputed = X[:, :non_empty_feature_count].copy()
        missing_mask = X[:, non_empty_feature_count:].astype(bool)

        n_features_original = len(self.statistics_)
        shape_original = (X.shape[0], n_features_original)
        X_original = np.zeros(shape_original)
        X_original[:, self.indicator_.features_] = missing_mask
        full_mask = X_original.astype(bool)

        imputed_idx, original_idx = 0, 0
        while imputed_idx < len(array_imputed.T):
            if not np.all(X_original[:, original_idx]):
                X_original[:, original_idx] = array_imputed.T[imputed_idx]
                imputed_idx += 1
                original_idx += 1
            else:
                original_idx += 1

        X_original[full_mask] = self.missing_values
        return X_original