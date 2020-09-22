
"""
Convert between NetSurfP-2.0 JSON output and other formats.

"""

import json
import argparse

from . import export_csv, export_npz


if __name__ == '__main__':
    parser = argparse.ArgumentParser('netsurfp2.convert', description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file (*.csv or *.npz)')
    args = parser.parse_args()

    with open(args.input) as fp:
        results = json.load(fp)

    print('Loaded {:,} results'.format(len(results)))

    output = args.output

    if output.lower().endswith('.csv'):
        with open(output, 'w') as fp:
            export_csv(results, fp)
    elif output.lower().endswith('.npz'):
        export_npz(results, output)
    else:
        with open(output, 'w') as fp:
            json.dump(results[:50], fp)
        print('Unknown output format (use .csv or .npz):', output)
