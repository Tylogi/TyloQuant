"""TQ checkpoint and random per-row precision contracts, without a GPU."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.tq_random import layout, random_tensor, row_bits
from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap, pack_tensor_payload, save, unpack_tensor_payload
from mfq.formats.tq import TQTensor
from mfq.formats.tq_rows import TQRows


class TqContainerTests(unittest.TestCase):
    def test_three_moe_banks_preserve_expert_rows(self):
        experts,hidden,width=3,11,37
        shapes={'gate':(experts,hidden,width),'up':(experts,hidden,width),
                'down':(experts,width,hidden)}
        tensors={f'experts.{role}.weight':random_tensor(e*n,k,e*n*256//2,seed=91+i,groupsize=16)
                 for i,(role,(e,n,k)) in enumerate(shapes.items())}
        logical={f'experts.{role}.weight':list(shape) for role,shape in shapes.items()}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'three-banks.mfq'
            save(path,FileHeader(model_arch='tq-test',extra={'tensor_shapes':logical}),tensors)
            with open_mmap(path) as model:
                self.assertEqual(set(model.records),set(tensors))
                for name,source in tensors.items():
                    self.assertEqual(model.records[name].dtype,'TQ')
                    self.assertEqual(model.header.extra['tensor_shapes'][name],logical[name])
                    loaded=model[name]
                    self.assertEqual(loaded.to_bytes(),source.to_bytes())
                    blob=loaded.to_bytes()
                    reader=TQRows(len(blob),lambda offset,size:blob[offset:offset+size])
                    _,rows,columns=logical[name]
                    expected=source.dequantize(columns).reshape(experts,rows,columns)
                    for expert in range(experts):
                        first,last=expert*rows,(expert+1)*rows
                        selected=reader.slice_rows(first,last)
                        np.testing.assert_array_equal(selected.bits,source.bits[first:last])
                        np.testing.assert_array_equal(selected.anchors,source.anchors[first:last])
                        np.testing.assert_array_equal(selected.dequantize(columns),expected[expert])

    def test_random_rows_match_bit_budget_without_sorting(self):
        for mean in (1, 1.1, 2.7, 4.5, 7.9, 8):
            total = round(1024 * mean)
            bits = row_bits(1024, total, 37)
            self.assertEqual(int(bits.sum()), total)
            self.assertTrue(np.all((bits >= 1) & (bits <= 8)))
            np.testing.assert_array_equal(bits, row_bits(1024, total, 37))
            if 1 < mean < 8:
                self.assertGreater(int(np.count_nonzero(bits[1:] < bits[:-1])), 0)

    def test_full_payload_size_includes_padding_and_metadata(self):
        info = layout(73, 641, 73 * 641 * 3 // 8)
        tensor = random_tensor(73, 641, info['target_bytes'], seed=51)
        self.assertEqual(tensor.nbytes, info['nbytes'])
        self.assertLessEqual(abs(tensor.nbytes - info['target_bytes']), info['stored_columns'] // 16)
        self.assertTrue(np.isfinite(tensor.dequantize(641)).all())

    def test_standalone_payload_and_mfq_container_preserve_rows(self):
        tensor = random_tensor(19, 257, 19 * 257 // 2, seed=11)
        dtype, payload = pack_tensor_payload(tensor)
        self.assertEqual(dtype, 'TQ')
        restored = unpack_tensor_payload(dtype, memoryview(payload))
        self.assertIsInstance(restored, TQTensor)
        self.assertEqual(payload, restored.to_bytes())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.mfq'
            save(path, FileHeader(model_arch='tq-test', extra={'tensor_shapes': {'weight': [19, 257]}}),
                 {'weight': tensor, 'norm': np.ones(19, np.float32)})
            with open_mmap(path) as model:
                self.assertEqual(model.records['weight'].dtype, 'TQ')
                self.assertEqual(model.header.extra['tensor_shapes']['weight'], [19, 257])
                self.assertEqual(model['weight'].to_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
