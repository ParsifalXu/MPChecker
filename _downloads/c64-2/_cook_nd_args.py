def _cook_nd_args(a, shapeless, s=None, axes=None, invreal=0):
    """
    Compute the 2-dimensional discrete Fourier Transform.

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    a : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    numpy.fft : Overall view of discrete Fourier transforms, with definitions
         and conventions used.
    ifft2 : The inverse two-dimensional FFT.
    fft : The one-dimensional FFT.
    fftn : The *n*-dimensional FFT.
    fftshift : Shifts zero-frequency terms to the center of the array.
        For two-dimensional input, swaps first and third quadrants, and second
        and fourth quadrants.

    Notes
    -----
    `fft2` is just `fftn` with a different default for `axes`.

    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.

    See `fftn` for details and a plotting example, and `numpy.fft` for
    definitions and conventions used.


    Examples
    --------
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.fft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ]])

    """
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        if shapeless == False:
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    if None in s:
        msg = ("Passing an array containing `None` values to `s` is "
               "deprecated in NumPy 2.0 and will raise an error in "
               "a future version of NumPy. To use the default behaviour "
               "of the corresponding 1-D transform, pass the value matching "
               "the default for its `n` parameter. To use the default "
               "behaviour for every axis, the `s` argument can be omitted.")
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    # use the whole input array along axis `i` if `s[i] == -1`
    s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
    return s, axes
