from contextlib import contextmanager


def is_buffer(obj, mode):
    return ("r" in mode and hasattr(obj, "read")) or (
        "w" in mode and hasattr(obj, "write")
    )


@contextmanager
def open_file(path_or_buf, mode="r"):
    if is_buffer(path_or_buf, mode):
        yield path_or_buf
    else:
        with open(path_or_buf, mode) as f:
            yield f
