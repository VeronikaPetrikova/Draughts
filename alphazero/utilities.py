#!/usr/bin/env python
#
# Created on: 2022-03-11
#

def histogram_to_string(histogram, eps=1e-8, separator=':'):
    return (
        f'{sum([v for k, v in histogram.items() if 1 <= k])}'
        f'{separator}{sum([v for k, v in histogram.items() if eps < k < 1])}'
        f'{separator}{sum([v for k, v in histogram.items() if abs(k) < eps])}'
        f'{separator}{sum([v for k, v in histogram.items() if -1 < k < -eps])}'
        f'{separator}{sum([v for k, v in histogram.items() if k <= -1])}'
    )
