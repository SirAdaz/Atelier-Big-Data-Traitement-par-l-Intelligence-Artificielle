class History:

    def __init__(self, line):
        line_split = line.split(",")
        self.date = line_split[0]
        self.hour = line_split[1]
        self.precision = line_split[2]