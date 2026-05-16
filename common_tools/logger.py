import os
from collections import OrderedDict

##TODO: generalize the logger to any data format
class SimpleLogger(object):
    def __init__(self, res_dir, kwarg_list, make_file=True):
        self.make_file = make_file
        self.res_dir = res_dir
        filename = "logger.txt" ## add datetime stamp
        if make_file:
            if not os.path.exists(self.res_dir):
                os.makedirs(self.res_dir)
            self.filename = os.path.join(self.res_dir, filename)
            with open(self.filename, 'w') as f:
                f.write("simple logger\n")

        self.kwarg_list = kwarg_list
        self.content = {}
        for kwarg in kwarg_list:
            self.content[kwarg] = []

    def logging(self, content):
        with open(self.filename, 'a') as f:
            if content is OrderedDict:
                for k, v in content.items():
                    f.write(k, v)
            else:
                f.write(content)

    def flush_out(self):
        with open(self.filename, 'a') as f:
            line = str()
            for k, v in self.content.items():
                line += (f"{k}: {v[-1]}; ")
            line += "\n"
            #print(line)
            f.write(line)

    def print_out(self, line):
        with open(self.filename, 'a') as f:
            f.write(line)

    def update(self, loss, kwarg:str):
        assert kwarg in self.kwarg_list
        self.content[kwarg].append(loss)

    def get_best(self, kwarg, best='min'):
        assert kwarg in self.kwarg_list
        if best == 'min':
            best_val = min(self.content[kwarg])
        else:
            best_val = max(self.content[kwarg])

        return self.content[kwarg].index(best_val)