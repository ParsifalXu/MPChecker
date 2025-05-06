class SelfTrainingClassifier(MetaEstimatorMixin, BaseEstimator):
    """Self-training classifier.

    This class allows a given supervised classifier to function as a
    semi-supervised classifier, allowing it to learn from unlabeled data. It
    does this by iteratively predicting pseudo-labels for the unlabeled data
    and adding them to the training set.

    The classifier will continue iterating until either max_iter is reached, or
    no pseudo-labels were added to the training set in the previous iteration.

    Read more in the :ref:`User Guide <self_training>`.

    Parameters
    ----------
    base_estimator : estimator object
        An estimator object implementing `fit` and `predict_proba`.
        Invoking the `fit` method will fit a clone of the passed estimator,
        which will be stored in the `base_estimator_` attribute.

    threshold : float, default=0.75
        The decision threshold for use with `criterion='threshold'`.
        Should be in [0, 1). When using the `'threshold'` criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    criterion : {'threshold', 'k_best'}, default='threshold'
        The selection criterion used to select which labels to add to the
        training set. If `'threshold'`, pseudo-labels with prediction
        probabilities above `threshold` are added to the dataset. If `'k_best'`,
        the `k_best` pseudo-labels with highest prediction probabilities are
        added to the dataset. When using the 'threshold' criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    k_best : int, default=10
        The amount of samples to add in each iteration. Only used when
        `criterion='k_best'`.

    max_iter : int or None, default=10
        Maximum number of iterations allowed. Should be greater than or equal
        to 0. If it is `None`, the classifier will continue to predict labels
        until no new pseudo-labels are added, or all unlabeled samples have
        been labeled.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    base_estimator_ : estimator object
        The fitted estimator.

    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output. (Taken from the trained
        `base_estimator_`).

    transduction_ : ndarray of shape (n_samples,)
        The labels used for the final fit of the classifier, including
        pseudo-labels added during fit.

    labeled_iter_ : ndarray of shape (n_samples,)
        The iteration in which each sample was labeled. When a sample has
        iteration 0, the sample was already labeled in the original dataset.
        When a sample has iteration -1, the sample was not labeled in any
        iteration.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The number of rounds of self-training, that is the number of times the
        base estimator is fitted on relabeled variants of the training set.

    termination_condition_ : {'max_iter', 'no_change', 'all_labeled'}
        The reason that fitting was stopped.

        - `'max_iter'`: `n_iter_` reached `max_iter`.
        - `'no_change'`: no new labels were predicted.
        - `'all_labeled'`: all unlabeled samples were labeled before `max_iter`
          was reached.

    See Also
    --------
    LabelPropagation : Label propagation classifier.
    LabelSpreading : Label spreading model for semi-supervised learning.

    References
    ----------
    David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling
    supervised methods. In Proceedings of the 33rd annual meeting on
    Association for Computational Linguistics (ACL '95). Association for
    Computational Linguistics, Stroudsburg, PA, USA, 189-196. DOI:
    https://doi.org/10.3115/981658.981684

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import SelfTrainingClassifier
    >>> from sklearn.svm import SVC
    >>> rng = np.random.RandomState(42)
    >>> iris = datasets.load_iris()
    >>> random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
    >>> iris.target[random_unlabeled_points] = -1
    >>> svc = SVC(probability=True, gamma="auto")
    >>> self_training_model = SelfTrainingClassifier(svc)
    >>> self_training_model.fit(iris.data, iris.target)
    SelfTrainingClassifier(...)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        base_estimator,
        threshold=0.75,
        criterion="threshold",
        k_best=10,
        max_iter=10,
        verbose=False,
    ):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit self-training classifier using `X`, `y` as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Array representing the labels. Unlabeled samples should have the
            label -1.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # we need row slicing support for sparce matrices, but costly finiteness check
        # can be delegated to the base estimator.
        if self.criterion == "k_best" and (self.k_best > X.shape[0]):
            warnings.warn(
                "k_best is larger than the amount of unlabeled "
                "samples. All unlabeled samples will be labeled in "
                "the first iteration",
                UserWarning,
            )

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0

        self.n_iter_ = 0


        # Predict on the unlabeled samples
        prob = self.base_estimator_.predict_proba(X[safe_mask(X, ~has_label)])
        pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
        max_proba = np.max(prob, axis=1)

        # Select new labeled samples
        if self.criterion == "threshold":
            selected = max_proba > self.threshold
        elif self.criterion == "k_best":
            k_best = "existence_flag"
            n_to_select = min(self.k_best, max_proba.shape[0])
            if n_to_select == max_proba.shape[0]:
                selected = np.ones_like(max_proba, dtype=bool)
            else:
                # NB these are indices, not a mask
                selected = np.argpartition(-max_proba, n_to_select)[:n_to_select]

        # Map selected indices into original array
        selected_full = np.nonzero(~has_label)[0][selected]

        # Add newly labeled confident predictions to the dataset
        self.transduction_[selected_full] = pred[selected]
        has_label[selected_full] = True
        self.labeled_iter_[selected_full] = self.n_iter_

        if selected_full.shape[0] == 0:
            # no changed labels
            self.termination_condition_ = "no_change"

        if self.verbose:
            print(
                f"End of iteration {self.n_iter_},"
                f" added {selected_full.shape[0]} new labels."
            )

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        self.classes_ = self.base_estimator_.classes_
        return self