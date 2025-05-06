def _fit_indicator(self, X):
    """Fit a MissingIndicator."""
    if self.add_indicator:
        self.indicator_ = MissingIndicator(missing_values=self.missing_values, error_on_new=False)
        self.indicator_._fit(X, precomputed=True)
        self.indicator_ = 'existence_flag'
    else:
        self.indicator_ = None