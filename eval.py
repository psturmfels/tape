#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from tape.main import run_eval
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run_eval())
