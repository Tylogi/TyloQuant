import unittest
import numpy as np

from bench.tq_random import random_tensor
from mfq.formats.tq import TQTensor
from mfq.formats.tq_rows import TQRows


class TestTQRows(unittest.TestCase):
    def test_odd_nibble_bounded_range(self):
        tensor = random_tensor(97, 640, 30000, seed=7, groupsize=256)
        payload = tensor.to_bytes()
        reads = []
        def read(offset, size):
            reads.append((offset, size))
            return payload[offset:offset+size]
        rows = TQRows(len(payload), read)
        selected = rows.slice_rows(3, 8)
        np.testing.assert_array_equal(selected.dequantize(), tensor.dequantize()[3:8])
        self.assertLess(sum(size for _, size in reads), len(payload)//4)
        np.testing.assert_array_equal(selected.bits, tensor.bits[3:8])

    def test_gather_preserves_order_and_duplicates(self):
        tensor = random_tensor(17, 513, 5000, seed=8, groupsize=256)
        payload = tensor.to_bytes()
        rows = TQRows(len(payload), lambda offset, size: payload[offset:offset+size])
        selected = [15, 1, 2, 1, 16, 3]
        np.testing.assert_array_equal(rows.gather_rows(selected).dequantize(513),
                                      tensor.dequantize(513)[selected])
        for indices in ([], [-1], [17]):
            with self.assertRaises(ValueError):
                rows.gather_rows(indices)

    def test_column_scales_and_truncation(self):
        source = random_tensor(9, 257, 2000, seed=9)
        tensor = TQTensor(source.rows, source.width, source.gs, True, source.descriptors,
                          np.linspace(.1, 1, source.width).astype(np.float16), source.packed)
        payload = tensor.to_bytes()
        rows = TQRows(len(payload), lambda offset, size: payload[offset:offset+size])
        np.testing.assert_array_equal(rows.slice_rows(2, 5).dequantize(), tensor.dequantize()[2:5])
        with self.assertRaises(ValueError):
            TQRows(len(payload)-1, lambda offset, size: payload[offset:offset+size])


if __name__ == '__main__':
    unittest.main()
