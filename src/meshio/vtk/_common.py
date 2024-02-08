from io import BytesIO, StringIO


def encode(string, binary):
    return string.encode() if binary else string


def tofile(arr, f, binary=True, sep=""):
    if binary:
        if isinstance(f, BytesIO):
            f.write(arr.tobytes())
        else:
            arr.tofile(f, sep=sep)

    else:
        if isinstance(f, StringIO):
            f.write(" ".join(str(x) for x in arr.ravel()))
        else:
            arr.tofile(f, sep=sep)
