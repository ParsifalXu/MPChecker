class HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting):
    """Histogram-based Gradient Boosting Regression Tree.

    This estimator is much faster than
    :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
    for big datasets (n_samples >= 10 000).

    This estimator has native support for missing values (NaNs). During
    training, the tree grower learns at each split point whether samples
    with missing values should go to the left or right child, based on the
    potential gain. When predicting, samples with missing values are
    assigned to the left or right child consequently. If no missing values
    were encountered for a given feature during training, then samples with
    missing values are mapped to whichever child has the most samples.

    This implementation is inspired by
    `LightGBM <https://github.com/Microsoft/LightGBM>`_.

    Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    loss : {'squared_error', 'least_squares', 'absolute_error', \
            'least_absolute_deviation', 'poisson'}, default='squared_error'
        The loss function to use in the boosting process. Note that the
        "least squares" and "poisson" losses actually implement
        "half least squares loss" and "half poisson deviance" to simplify the
        computation of the gradient. Furthermore, "poisson" loss internally
        uses a log-link and requires ``y >= 0``.

        .. versionchanged:: 0.23
           Added option 'poisson'.

        .. deprecated:: 1.0
            The loss 'least_squares' was deprecated in v1.0 and will be removed
            in version 1.2. Use `loss='squared_error'` which is equivalent.

        .. deprecated:: 1.0
            The loss 'least_absolute_deviation' was deprecated in v1.0 and will
            be removed in version 1.2. Use `loss='absolute_error'` which is
            equivalent.

    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    l2_regularization : float, default=0
        The L2 regularization parameter. Use ``0`` for no regularization
        (default).
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    categorical_features : array-like of {bool, int} of shape (n_features) \
            or shape (n_categorical_features,), default=None
        Indicates the categorical features.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

        Read more in the :ref:`User Guide <categorical_support_gbdt>`.

        .. versionadded:: 0.24

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonic constraint to enforce on each feature. -1, 1
        and 0 respectively correspond to a negative constraint, positive
        constraint and no constraint. Read more in the :ref:`User Guide
        <monotonic_cst_gbdt>`.

        .. versionadded:: 0.23

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.
    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.

        .. versionadded:: 0.23

    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used. If
        ``scoring='loss'``, early stopping is checked w.r.t the loss value.
        Only used if early stopping is performed.
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if early stopping is performed.
    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. Only used if early stopping is performed.
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores during early
        stopping. The higher the tolerance, the more likely we are to early
        stop: higher tolerance means that it will be harder for subsequent
        iterations to be considered an improvement upon the reference score.
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    do_early_stopping_ : bool
        Indicates whether early stopping is used during training.
    n_iter_ : int
        The number of iterations as selected by early stopping, depending on
        the `early_stopping` parameter. Otherwise it corresponds to max_iter.
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. For regressors,
        this is always 1.
    train_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    validation_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the held-out validation data. The
        first entry is the score of the ensemble before the first iteration.
        Scores are computed according to the ``scoring`` parameter. Empty if
        no early stopping or if ``validation_fraction`` is None.
    is_categorical_ : ndarray, shape (n_features, ) or None
        Boolean mask for the categorical features. ``None`` if there are no
        categorical features.
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    See Also
    --------
    GradientBoostingRegressor : Exact gradient boosting method that does not
        scale as good on datasets with a large number of samples.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    RandomForestRegressor : A meta-estimator that fits a number of decision
        tree regressors on various sub-samples of the dataset and uses
        averaging to improve the statistical performance and control
        over-fitting.
    AdaBoostRegressor : A meta-estimator that begins by fitting a regressor
        on the original dataset and then fits additional copies of the
        regressor on the same dataset but where the weights of instances are
        adjusted according to the error of the current prediction. As such,
        subsequent regressors focus more on difficult cases.

    Examples
    --------
    >>> from sklearn.ensemble import HistGradientBoostingRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> est = HistGradientBoostingRegressor().fit(X, y)
    >>> est.score(X, y)
    0.92...
    """

    _VALID_LOSSES = (
        "squared_error",
        "least_squares",
        "absolute_error",
        "least_absolute_deviation",
        "poisson",
    )

    def __init__(
        self,
        loss="squared_error",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
    ):
        super(HistGradientBoostingRegressor, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            monotonic_cst=monotonic_cst,
            categorical_features=categorical_features,
            early_stopping=early_stopping,
            warm_start=warm_start,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

    def _encode_y(self, y):
        # Just convert y to the expected dtype
        self.n_trees_per_iteration_ = 1
        if self.loss == "poisson":
            # Ensure y >= 0 and sum(y) > 0
            if y < 0:
                raise ValueError(
                    "loss='poisson' requires non-negative y and sum(y) > 0."
                )
        return y
