import re
class pmkwd_line(object):
    """A valid line from permas-dat/dato/post file in the $STRUCTURE block"""
    # static variables
    ## re to split a permas keyword line
    splkwd_pattern =  '[A-Za-z0-9_]+(?:\s*=\s*)?[A-Za-z0-9_]+'
    splkwd         = re.compile(splkwd_pattern)
    def __init__(self,location,string):
        self.location = location
        self.kwds     = type(self).splkwd.findall(string)
        
