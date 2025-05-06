class DataFrame(NDFrame, OpsMixin):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Data structure also contains labeled axes (rows and columns).
    Arithmetic operations align on both row and column labels. Can be
    thought of as a dict-like container for Series objects. The primary
    pandas data structure.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Dict can contain Series, arrays, constants, dataclass or list-like objects. If
        data is a dict, column order follows insertion-order. If a dict contains Series
        which have an index defined, it is aligned by its index. This alignment also
        occurs if data is a Series or a DataFrame itself. Alignment is done on
        Series/DataFrame inputs.

        .. versionchanged:: 0.25.0
           If data is a list of dicts, column order follows insertion-order.

    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    columns : Index or array-like
        Column labels to use for resulting frame when data does not have them,
        defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
        will perform column selection instead.
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer.
    copy : bool or None, default None
        Copy data from inputs.
        For dict data, the default of None behaves like ``copy=True``.  For DataFrame
        or 2d ndarray input, the default of None behaves like ``copy=False``.
        If data is a dict containing one or more Series (possibly of different dtypes),
        ``copy=False`` will ensure that these inputs are not copied.

        .. versionchanged:: 1.3.0

    See Also
    --------
    DataFrame.from_records : Constructor from tuples, also record arrays.
    DataFrame.from_dict : From dicts of Series, arrays, or dicts.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_table : Read general delimited file into DataFrame.
    read_clipboard : Read text from clipboard into DataFrame.

    Notes
    -----
    Please reference the :ref:`User Guide <basics.dataframe>` for more information.

    Examples
    --------
    Constructing DataFrame from a dictionary.

    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data=d)
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Notice that the inferred dtype is int64.

    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    To enforce a single dtype:

    >>> df = pd.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object

    Constructing DataFrame from a dictionary including Series:

    >>> d = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}
    >>> pd.DataFrame(data=d, index=[0, 1, 2, 3])
       col1  col2
    0     0   NaN
    1     1   NaN
    2     2   2.0
    3     3   3.0

    Constructing DataFrame from numpy ndarray:

    >>> df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ...                    columns=['a', 'b', 'c'])
    >>> df2
       a  b  c
    0  1  2  3
    1  4  5  6
    2  7  8  9

    Constructing DataFrame from a numpy ndarray that has labeled columns:

    >>> data = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    ...                 dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")])
    >>> df3 = pd.DataFrame(data, columns=['c', 'a'])
    ...
    >>> df3
       c  a
    0  3  1
    1  6  4
    2  9  7

    Constructing DataFrame from dataclass:

    >>> from dataclasses import make_dataclass
    >>> Point = make_dataclass("Point", [("x", int), ("y", int)])
    >>> pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])
       x  y
    0  0  0
    1  0  3
    2  2  3

    Constructing DataFrame from Series/DataFrame:

    >>> ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
    >>> df = pd.DataFrame(data=ser, index=["a", "c"])
    >>> df
       0
    a  1
    c  3

    >>> df1 = pd.DataFrame([1, 2, 3], index=["a", "b", "c"], columns=["x"])
    >>> df2 = pd.DataFrame(data=df1, index=["a", "c"])
    >>> df2
       x
    a  1
    c  3
    """
    @classmethod
    def from_dict(
        cls,
        data: dict,
        orient: str = "columns",
        dtype: Dtype | None = None,
        columns: Axes | None = None,
    ) -> DataFrame:
        """
        Construct DataFrame from dict of array-like or dicts.

        Creates DataFrame object from dictionary by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {'columns', 'index', 'tight'}, default 'columns'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
            If 'tight', assume a dict with keys ['index', 'columns', 'data',
            'index_names', 'column_names'].

            .. versionadded:: 1.4.0
               'tight' as an allowed value for the ``orient`` argument

        dtype : dtype, default None
            Data type to force after DataFrame construction, otherwise infer.
        columns : list, default None
            Column labels to use when ``orient='index'``. Raises a ValueError
            if used with ``orient='columns'`` or ``orient='tight'``.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_records : DataFrame from structured ndarray, sequence
            of tuples or dicts, or DataFrame.
        DataFrame : DataFrame object creation using constructor.
        DataFrame.to_dict : Convert the DataFrame to a dictionary.

        Examples
        --------
        By default the keys of the dict become the DataFrame columns:

        >>> data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
        >>> pd.DataFrame.from_dict(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Specify ``orient='index'`` to create the DataFrame using dictionary
        keys as rows:

        >>> data = {'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}
        >>> pd.DataFrame.from_dict(data, orient='index')
               0  1  2  3
        row_1  3  2  1  0
        row_2  a  b  c  d

        When using the 'index' orientation, the column names can be
        specified manually:

        >>> pd.DataFrame.from_dict(data, orient='index',
        ...                        columns=['A', 'B', 'C', 'D'])
               A  B  C  D
        row_1  3  2  1  0
        row_2  a  b  c  d

        Specify ``orient='tight'`` to create the DataFrame using a 'tight'
        format:

        >>> data = {'index': [('a', 'b'), ('a', 'c')],
        ...         'columns': [('x', 1), ('y', 2)],
        ...         'data': [[1, 3], [2, 4]],
        ...         'index_names': ['n1', 'n2'],
        ...         'column_names': ['z1', 'z2']}
        >>> pd.DataFrame.from_dict(data, orient='tight')
        z1     x  y
        z2     1  2
        n1 n2
        a  b   1  3
           c   2  4
        """
        index = None
        if orient == "index":
            if len(data) > 0:
                # TODO speed up Series case
                if isinstance(list(data.values())[0], (Series, dict)):
                    data = _from_nested_dict(data)
                else:
                    index = list(data.keys())
                    # error: Incompatible types in assignment (expression has type
                    # "List[Any]", variable has type "Dict[Any, Any]")
                    data = list(data.values())  # type: ignore[assignment]
        elif orient == "columns" or orient == "tight":
            if columns is not None:
                raise ValueError(f"cannot use columns parameter with orient='{orient}'")
        else:  # pragma: no cover
            raise ValueError(
                f"Expected 'index', 'columns' or 'tight' for orient parameter. "
                f"Got '{orient}' instead"
            )

        if orient != "tight":
            return cls(data, index=index, columns=columns, dtype=dtype)
        else:
            realdata = data["data"]

            def create_index(indexlist, namelist):
                index: Index
                if len(namelist) > 1:
                    index = MultiIndex.from_tuples(indexlist, names=namelist)
                else:
                    index = Index(indexlist, name=namelist[0])
                return index

            index = create_index(data["index"], data["index_names"])
            columns = create_index(data["columns"], data["column_names"])
            return cls(realdata, index=index, columns=columns, dtype=dtype)
