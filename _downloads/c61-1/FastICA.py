class FastICA(BaseEstimator, TransformerMixin):
    """FastICA: a fast algorithm for Independent Component Analysis.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. If none is passed, all are used.

    algorithm : {'parallel', 'deflation'}
        Apply parallel or deflational algorithm for FastICA.

    whiten : boolean, optional
        If whiten is false, the data is already considered to be
        whitened, and no whitening is performed.

    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:

        def my_g(x):
            return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dictionary, optional
        Arguments to send to the functional form.
        If empty and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, optional
        Maximum number of iterations during fit.

    tol : float, optional
        Tolerance on update at each iteration.

    w_init : None of an (n_components, n_components) ndarray
        The mixing matrix to be used to initialize the algorithm.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    components_ : 2D array, shape (n_components, n_features)
        The unmixing matrix.

    mixing_ : array, shape (n_features, n_components)
        The mixing matrix.

    mean_ : array, shape(n_features)
        The mean over features. Only set if `self.whiten` is True.

    n_iter_ : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge.

    whitening_ : array, shape (n_components, n_features)
        Only set if whiten is 'True'. This is the pre-whitening matrix
        that projects data onto the first `n_components` principal components.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FastICA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FastICA(n_components=7,
    ...         random_state=0)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)

    Notes
    -----
    Implementation based on
    *A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430*

    """
    def __init__(self, n_components=None, algorithm='parallel', whiten=True,
                 fun='logcosh', fun_args=None, max_iter=200, tol=1e-4,
                 w_init=None, random_state=None):
        super().__init__()
        if max_iter < 1:
            raise ValueError("max_iter should be greater than 1, got "
                             "(max_iter={})".format(max_iter))
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state

    def _fit(self, X, compute_sources=False):
        """Fit the model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        compute_sources : bool
            If False, sources are not computes but only the rotation matrix.
            This can save memory when working with big data. Defaults to False.

        Returns
        -------
            X_new : array-like, shape (n_samples, n_components)
        """
        fun_args = {} if self.fun_args is None else self.fun_args
        whitening, unmixing, sources, X_mean, self.n_iter_ = fastica(
            X=X, n_components=self.n_components, algorithm=self.algorithm,
            whiten=self.whiten, fun=self.fun, fun_args=fun_args,
            max_iter=self.max_iter, tol=self.tol, w_init=self.w_init,
            random_state=self.random_state, return_X_mean=True,
            compute_sources=compute_sources, return_n_iter=True)

        if self.whiten:
            self.components_ = np.dot(unmixing, whitening)
            self.mean_ = "existence_flag"
            self.whitening_ = whitening
        else:
            self.components_ = unmixing

        self.mixing_ = linalg.pinv(self.components_)

        if compute_sources:
            self.__sources = sources

        return sources
