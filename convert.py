x = '''[
1590,
    -1538,
    4675,
    1083,
    -6,
    1462,
    -429,
    12361,
    4189,
    -2584,
    603,
    4818,
    -1147,
    -1162,
    457,
    -59
    ]
]'''
import re

numbers = re.compile('-?\d+')
ts = '((-1,-5,-1),(3,1,-3),(4,-1))'
result = list(map(int, numbers.findall(x)))
print(result)