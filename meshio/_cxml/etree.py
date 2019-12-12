# class etree


class Element:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._children = []
        self.text = None
        self.text_writer = None

    def insert(self, pos, elem):
        self._children.insert(pos, elem)

    def set(self, key, value):
        self.kwargs[key] = value

    def write(self, f):
        kw_list = ['{}="{}"'.format(key, value) for key, value in self.kwargs.items()]
        f.write("<{}>\n".format(" ".join([self.name] + kw_list)))
        if self.text:
            f.write(self.text)
            f.write("\n")
        if self.text_writer:
            self.text_writer(f)
            f.write("\n")
        for child in self._children:
            child.write(f)
        f.write("</{}>\n".format(self.name))


class SubElement(Element):
    def __init__(self, parent, name, **kwargs):
        super().__init__(name, **kwargs)
        parent._children.append(self)


class Comment:
    def __init__(self, text):
        self.text = text

    def write(self, f):
        f.write("<!--{}-->\n".format(self.text))


class ElementTree:
    def __init__(self, root):
        self.root = root

    def write(self, filename):
        with open(filename, "w") as f:
            self.root.write(f)
