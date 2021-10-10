from itertools import zip_longest
import numpy as np
import pprint

"""
input: "1101101"
output: [1, 1, -1, 1, 1, -1, 1]
"""
def bit_converter(strings):
    bits = []
    for char in strings:
        if char == "0":
            bits.append(-1)
        else:
            bits.append(1)
    return np.array(bits)

codes = np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]])
series = np.array([*map(bit_converter, zip_longest(*["1101101" , "011" , "110100010" , "00000000000010"], fillvalue="0"))])

channels = []
for data in series:
    channel = np.array([0 for _ in range(4)])
    for code, datum in zip_longest(codes, data):
        spread = code * datum
        channel += spread
    channels.append(np.array(channel))

results = []
for channel in channels:
    result = []
    for code in codes:
        result.append(code@channel/4)
    results.append(result)
pprint.pprint(results)
