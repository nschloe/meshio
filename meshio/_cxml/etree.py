# This XML writer is a drop-in replacement for LXML/Python XML Etree. It only offers one
# other member: self.text_write.
# The problem is that, for LXML, the entire etree has to be constructed in memory before
# writing it to a file. Many mesh formats that use XML have lots of int or float data
# written in the text fields. Converting this to ASCII first requires a lot of memory.
# This etree here allows the writing method to write to the file directly, without
# having to create a string representation first.


class Element:
    def __init__(self, name, **kwargs):
        self.name = name
        self.attrib = kwargs
        self._children = []
        self.text = None
        self.text_writer = None

    def insert(self, pos, elem):
        self._children.insert(pos, elem)

    def set(self, key, value):
        self.attrib[key] = value

    def write(self, f):
        kw_list = [f'{key}="{value}"' for key, value in self.attrib.items()]
        f.write("<{}>\n".format(" ".join([self.name] + kw_list)))
        if self.text:
            f.write(self.text)
            f.write("\n")
        if self.text_writer:
            self.text_writer(f)
            f.write("\n")
        for child in self._children:
            child.write(f)
        f.write(f"</{self.name}>\n")


class SubElement(Element):
    def __init__(self, parent, name, **kwargs):
        super().__init__(name, **kwargs)
        parent._children.append(self)


class Comment:
    def __init__(self, text):
        self.text = text

    def write(self, f):
        f.write(f"<!--{self.text}-->\n")


class ElementTree:
    def __init__(self, root):
        self.root = root

    def write(self, filename, xml_declaration=True):
        with open(filename, "w") as f:
            if xml_declaration:
                f.write('<?xml version="1.0"?>\n')
            self.root.write(f)
