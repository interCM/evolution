class Monitor(object):
    """ A class to record events in the course of evolution.
    """
    def __init__(self, log_file_name):
        self.log_file = open(log_file_name, 'w')
        self.pop_name = ""

    def write(self, tag, msg):
        self.log_file.write("%s\t%s\t%s\n" % (self.pop_name, tag, msg))
        self.log_file.flush()

    def close(self):
        self.log_file.close()
