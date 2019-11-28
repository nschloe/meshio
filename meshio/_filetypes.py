from pathlib import Path

from ._exceptions import ReadError, WriteError

_name_to_reader = dict()
_ext_to_reader = dict()

_name_to_writer = dict()
_ext_to_writer = dict()


def register_reader(name, fn, *extensions):
    _name_to_reader[name] = fn
    for ext in extensions:
        _ext_to_reader[ext] = fn


def register_writer(name, fn, *extensions):
    _name_to_writer[name] = fn
    for ext in extensions:
        _ext_to_writer[ext] = fn


null = object()


def dict_fallthrough(key, dicts, default=null):
    for d in dicts:
        try:
            return d[key]
        except KeyError:
            pass
    if default is null:
        raise KeyError("Key does not exist in any of these dicts")
    else:
        return default


def get_reader(name_or_ext):
    if name_or_ext is None:
        raise ReadError("File format must be given")

    reader = dict_fallthrough(name_or_ext, [_name_to_reader, _ext_to_reader], None)
    if reader:
        return reader

    ext = ""
    for suffix in reversed(Path(name_or_ext).suffixes):
        ext = suffix + ext
        try:
            return _ext_to_reader[ext]
        except KeyError:
            pass

    raise ReadError(
        "Name or extension not found. "
        "For reading, valid names are \n\t{}\nand extensions\n\t{}".format(
            "\n\t".join(sorted(_name_to_reader)), "\n\t".join(sorted(_ext_to_reader)),
        )
    )


def get_writer(name_or_ext):
    if name_or_ext is None:
        raise WriteError("File format must be given")

    out = dict_fallthrough(name_or_ext, [_name_to_writer, _ext_to_writer], None)
    if out:
        return out

    ext = ""
    for suffix in reversed(Path(name_or_ext).suffixes):
        ext = suffix + ext
        try:
            return _ext_to_writer[ext]
        except KeyError:
            pass

    raise WriteError(
        "Name or extension not found. "
        "For writing, valid names are \n\t{}\nand extensions are\n\t{}".format(
            "\n\t".join(sorted(_name_to_writer)), "\n\t".join(sorted(_ext_to_writer)),
        )
    )


def revpartial(func, *args, **kwargs):
    """
    Like functools.partial, but positional arguments passed to the partial are
    prepended rather than appended to those passed when constructing it.
    """
    # TODO: replace with functools.partial? Positional arguments are never used
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**kwargs, **fkeywords}
        return func(*fargs, *args, **newkeywords)

    return newfunc
