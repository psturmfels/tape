#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from tape.main import run_train
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run_train())
