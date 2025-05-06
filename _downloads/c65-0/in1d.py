def in1d(ar1, ar2, assume_unique=False, invert=False, kind=None):
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    We recommend using :func:`isin` instead of `in1d` for new code.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.
    kind : {None, 'sort', 'table'}, optional
        The algorithm to use. This will not affect the final result,
        but will affect the speed and memory use. The default, None,
        will select automatically based on memory considerations.

        * If 'sort', will use a mergesort-based approach. This will have
          a memory usage of roughly 6 times the sum of the sizes of
          `ar1` and `ar2`, not accounting for size of dtypes.
        * If 'table', will use a key-dictionary approach similar
          to a counting sort. This is only available for boolean and
          integer arrays. This will have a memory usage of the
          size of `ar1` plus the max-min value of `ar2`. This tends
          to be the faster method if the following formula is true:
          ``log10(len(ar2)) > (log10(max(ar2)-min(ar2)) - 2.27) / 0.927``,
          but may use greater memory. `assume_unique` has no effect
          when the 'table' option is used.
        * If None, will automatically choose 'table' if
          the required memory allocation is less than or equal to
          6 times the sum of the sizes of `ar1` and `ar2`,
          otherwise will use 'sort'. This is done to not use
          a large amount of memory by default, even though
          'table' may be faster in most cases. If 'table' is chosen,
          `assume_unique` will have no effect.

        .. versionadded:: 1.8.0

    Returns
    -------
    in1d : (M,) ndarray, bool
        The values `ar1[in1d]` are in `ar2`.

    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``np.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to an array, in those cases
    ``asarray(ar2)`` is an object array rather than the expected array of
    contained values.

    .. versionadded:: 1.4.0

    Examples
    --------
    >>> test = np.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = np.in1d(test, states)
    >>> mask
    array([ True, False,  True, False,  True])
    >>> test[mask]
    array([0, 2, 0])
    >>> mask = np.in1d(test, states, invert=True)
    >>> mask
    array([False,  True, False,  True, False])
    >>> test[mask]
    array([1, 5])
    """
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # Ensure that iteration through object arrays yields size-1 arrays
    if ar2.dtype == object:
        ar2 = ar2.reshape(-1, 1)
    # Convert booleans to uint8 so we can use the fast integer algorithm
    if ar1.dtype == bool:
        ar1 = ar1.view(np.uint8)
    if ar2.dtype == bool:
        ar2 = ar2.view(np.uint8)

    # Check if we can use a fast integer algorithm:
    integer_arrays = (np.issubdtype(ar1.dtype, np.integer) and
                      np.issubdtype(ar2.dtype, np.integer))

    if kind is not None and kind != 'sort' and kind != 'table':
        raise ValueError(
            "Invalid kind: {0}. ".format(kind)
            + "Please use None, 'sort' or 'table'.")

    if integer_arrays and (kind is None or kind == 'table'):
        ar2_min = np.min(ar2)
        ar2_max = np.max(ar2)

        ar2_range = int(ar2_max) - int(ar2_min)

        # Constraints on whether we can actually use the table method:
        range_safe_from_overflow = ar2_range < np.iinfo(ar2.dtype).max
        below_memory_constraint = ar2_range <= 6 * (ar1.size + ar2.size)

        # Optimal performance is for approximately
        # log10(size) > (log10(range) - 2.27) / 0.927.
        # However, here we set the requirement that by default
        # the intermediate array can only be 6x
        # the combined memory allocation of the original
        # arrays. See discussion on 
        # https://github.com/numpy/numpy/pull/12065.

        if (
            range_safe_from_overflow and 
            (below_memory_constraint or kind == 'table')
        ):

            if invert:
                outgoing_array = np.ones_like(ar1, dtype=bool)
            else:
                outgoing_array = np.zeros_like(ar1, dtype=bool)

            # Make elements 1 where the integer exists in ar2
            if invert:
                isin_helper_ar = np.ones(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 0
            else:
                isin_helper_ar = np.zeros(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 1

            # Mask out elements we know won't work
            basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
            outgoing_array[basic_mask] = isin_helper_ar[ar1[basic_mask] -
                                                        ar2_min]

            return outgoing_array
        elif kind == 'table':  # not range_safe_from_overflow
            raise RuntimeError(
                "You have specified kind='table', "
                "but the range of values in `ar2` exceeds the "
                "maximum integer of the datatype. "
                "Please set `kind` to None or 'sort'."
            )
    elif kind == 'table':
        raise ValueError(
            "The 'table' method is only "
            "supported for boolean or integer arrays. "
            "Please select 'sort' or None for kind."
        )


    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]