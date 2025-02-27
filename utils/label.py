class Label:
    def __init__(self, s, e, p):
        self.start = s # start time
        self.end = e # end time
        self.phone = p # phoneme

    def length(self): # label length
        l = self.end - self.start
        if l < 0:
            logging.warning('Negative length.')
        return l
    
    def __sub__(self, other):
        return Label(self.start - other, self.end - other, self.phone)
    
    def __add__(self, other):
        return Label(self.start + other, self.end + other, self.phone)