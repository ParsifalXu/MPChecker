class OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
    """Orthogonal Matching Pursuit model (OMP).

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : 'auto' or bool, default='auto'
        Whether to use a precomputed Gram and Xy matrix to speed up
        calculations. Improves performance when :term:`n_targets` or
        :term:`n_samples` is very large. Note that if you already have such
        matrices, you can pass them directly to the fit method.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formula).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int or array-like
        Number of active features across every target.

    n_nonzero_coefs_ : int
        The number of non-zero coefficients in the solution. If
        `n_nonzero_coefs` is None and `tol` is None this value is either set
        to 10% of `n_features` or 1, whichever is greater.

    Examples
    --------
    >>> from sklearn.linear_model import OrthogonalMatchingPursuit
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> reg = OrthogonalMatchingPursuit().fit(X, y)
    >>> reg.score(X, y)
    0.9991...
    >>> reg.predict(X[:1,])
    array([-78.3854...])

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    See Also
    --------
    orthogonal_mp
    orthogonal_mp_gram
    lars_path
    Lars
    LassoLars
    sklearn.decomposition.sparse_encode
    OrthogonalMatchingPursuitCV
    """
    @_deprecate_positional_args
    def __init__(self, *, n_nonzero_coefs=None, tol=None, fit_intercept=True,
                 normalize=True, precompute='auto'):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute

    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary


        Returns
        -------
        self : object
            returns an instance of self.
        """
        X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)
        n_features = X.shape[1]

        X, y, X_offset, y_offset, X_scale, Gram, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=True)

        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.n_nonzero_coefs is None and self.tol is None:
            # default for n_nonzero_coefs is 0.1 * n_features
            # but at least one.
            self.n_nonzero_coefs_ = max(int(0.1 * n_features), 1)
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs

        if Gram is False:
            coef_, self.n_iter_ = orthogonal_mp(
                X, y, n_nonzero_coefs=self.n_nonzero_coefs_, tol=self.tol,
                precompute=False, copy_X=True,
                return_n_iter=True)
        else:
            norms_sq = np.sum(y ** 2, axis=0) if self.tol is not None else None

            coef_, self.n_iter_ = orthogonal_mp_gram(
                Gram, Xy=Xy, n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol, norms_squared=norms_sq,
                copy_Gram=True, copy_Xy=True,
                return_n_iter=True)
        self.coef_ = coef_.T
        self._set_intercept(X_offset, y_offset, X_scale)
        return self