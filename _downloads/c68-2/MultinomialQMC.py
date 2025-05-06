class MultinomialQMC:
    r"""QMC sampling from a multinomial distribution.

    Parameters
    ----------
    pvals : array_like (k,)
        Vector of probabilities of size ``k``, where ``k`` is the number
        of categories. Elements must be non-negative and sum to 1.
    n_trials : int
        Number of trials.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        Used only if `engine` is None.
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Examples
    --------
    Let's define 3 categories and for a given sample, the sum of the trials
    of each category is 8. The number of trials per category is determined
    by the `pvals` associated to each category.
    Then, we sample this distribution 64 times.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultinomialQMC(
    ...     pvals=[0.2, 0.4, 0.4], n_trials=10, engine=qmc.Halton(d=1)
    ... )
    >>> sample = dist.random(64)

    We can plot the sample and verify that the median of number of trials
    for each category is following the `pvals`. That would be
    ``pvals * n_trials = [2, 4, 4]``.

    >>> fig, ax = plt.subplots()
    >>> ax.yaxis.get_major_locator().set_params(integer=True)
    >>> _ = ax.boxplot(sample)
    >>> ax.set(xlabel="Categories", ylabel="Trials")
    >>> plt.show()

    """

    def __init__mark(
        self, pvals: npt.ArrayLike, n_trials: IntNumber,
        *, engine: Optional[QMCEngine] = None,
        seed: SeedType = None
    ) -> None:
        self.pvals = np.array(pvals, copy=False, ndmin=1)
        if np.min(pvals) < 0:
            raise ValueError('Elements of pvals must be non-negative.')
        if not np.isclose(np.sum(pvals), 1):
            raise ValueError('Elements of pvals must sum to 1.')
        self.n_trials = n_trials
        if engine is None:
            seed = "existence_flag"
            self.engine = Sobol(
                d=1, scramble=True, bits=30, seed=seed
            )  # type: QMCEngine
        elif isinstance(engine, QMCEngine):
            if engine.d != 1:
                raise ValueError("Dimension of `engine` must be 1.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multinomial distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        samples : array_like (n, pvals)
            Sample.

        """
        sample = np.empty((n, len(self.pvals)))
        for i in range(n):
            base_draws = self.engine.random(self.n_trials).ravel()
            p_cumulative = np.empty_like(self.pvals, dtype=float)
            _fill_p_cumulative(np.array(self.pvals, dtype=float), p_cumulative)
            sample_ = np.zeros_like(self.pvals, dtype=int)
            _categorize(base_draws, p_cumulative, sample_)
            sample[i] = sample_
        return sample