from pathlib import Path
import tempfile
import unittest

import numpy as np

from bench.generate_tq_fixture import generate, histogram
from bench.tq_fixture import TQFixture, Segment
from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap, save
from mfq.formats.nint import NintSpec, NintTensor


class TestTQFixture(unittest.TestCase):
    def test_histogram_exact_budget(self):
        for rows in (1,19,4096):
            for budget in (rows,rows*2+rows//3,rows*8):
                counts=histogram(rows,budget)
                self.assertEqual(sum(counts),rows)
                self.assertEqual(sum((i+1)*n for i,n in enumerate(counts)),budget)

    def test_generation_and_bounded_projection(self):
        rows,columns=137,512
        shape=(rows,columns)
        reference=NintTensor(NintSpec(2,32,4),shape,-1,
            np.zeros((rows,columns//32,32),np.uint8),np.ones(rows,np.float32),np.zeros(rows,np.float32),
            np.ones((rows,columns//32),np.uint8),np.zeros((rows,columns//32),np.uint8),columns)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save(root/'reference.mfq',FileHeader(model_arch='fixture'),{'linear.weight':reference})
            result=generate(root/'reference.mfq',root/'test.tq-fixture.json')
            self.assertLess(abs(result['tq_bytes']-result['original_bytes']),columns//8)
            with TQFixture(root/'test.tq-fixture.json') as fixture:
                entire=fixture.rows('linear.weight',0,rows)
                selected=fixture.rows('linear.weight',7,21)
                np.testing.assert_array_equal(selected.dequantize(),entire.dequantize()[7:21])
                np.testing.assert_array_equal(selected.bits,entire.bits[7:21])
                self.assertGreater(len(np.unique(entire.bits)),3)
                spec=result['tensors']['linear.weight']['segments'][0]
                segment=Segment(spec,fixture.read_file)
                # Unaligned byte reads must agree with the ordinary payload.
                self.assertEqual(segment.read(3,spec['nbytes']-7),entire.to_bytes()[3:-4])

    def test_combined_experts_generate_separate_gate_up_banks(self):
        experts,hidden,columns=3,5,256
        shape=(experts,2*hidden,columns)
        reference=NintTensor(NintSpec(2,32,4),shape,-1,
            np.zeros((experts*2*hidden,columns//32,32),np.uint8),
            np.ones(experts*2*hidden,np.float32),np.zeros(experts*2*hidden,np.float32),
            np.ones((experts*2*hidden,columns//32),np.uint8),
            np.zeros((experts*2*hidden,columns//32),np.uint8),columns)
        name='predictor.block.0.mlp.experts.gate_up.weight'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save(root/'reference.mfq',FileHeader(model_arch='fixture'),{name:reference})
            result=generate(root/'reference.mfq',root/'test.tq-fixture.json')
            self.assertEqual(result['remove_tensors'],[name])
            self.assertNotIn(name,result['tensors'])
            with open_mmap(root/'reference.mfq') as original:
                self.assertEqual(sum(t['original_bytes'] for t in result['tensors'].values()),
                                 original.records[name].nbytes)
            with TQFixture(root/'test.tq-fixture.json') as fixture:
                for role in ('gate','up'):
                    key=name.replace('gate_up.weight',role+'.weight')
                    self.assertEqual(result['tensors'][key]['shape'],[experts,hidden,columns])
                    self.assertEqual(fixture.rows(key,hidden,2*hidden).dequantize().shape,(hidden,columns))


if __name__ == '__main__':
    unittest.main()
