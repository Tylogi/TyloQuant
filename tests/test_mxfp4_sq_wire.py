import re
import struct
import unittest
import importlib.util
import sys
import types
from unittest import mock

from bench.cuda_mxfp4_sq_fixtures import ROOT, fixture, pack, palette_nibbles


class SqWireTest(unittest.TestCase):
    def test_torch_loader_uses_native_header_language_standard(self):
        spec = importlib.util.spec_from_file_location("sq_test_ext", ROOT / "mfq/kernels/cuda/_ext.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extension = types.ModuleType("torch.utils.cpp_extension")
        extension.load = mock.Mock(return_value=object())
        with mock.patch.object(module, "_ensure_msvc"), mock.patch.dict(sys.modules, {
            "torch": types.ModuleType("torch"),
            "torch.utils": types.ModuleType("torch.utils"),
            "torch.utils.cpp_extension": extension,
        }):
            module.ext()
        kwargs = extension.load.call_args.kwargs
        self.assertTrue(any("std:c++20" in flag or "std=c++20" in flag for flag in kwargs["extra_cflags"]))
        self.assertIn("-std=c++20", kwargs["extra_cuda_cflags"])

    def test_frozen_palette_parity(self):
        source = (ROOT / "mfq/kernels/cuda/mxfp4_sq.cu").read_text()
        for bits in (2, 3):
            match = re.search(rf"kSq{bits}Palette\[\d+\]\s*=\s*\{{(.*?)\}};", source, re.S)
            self.assertIsNotNone(match)
            self.assertEqual([int(x) for x in re.findall(r"\d+", match.group(1))], palette_nibbles(bits))

    def test_little_endian_cross_byte_pack(self):
        self.assertEqual(pack([0, 1, 2, 3], 2), bytes([0xe4]))
        for bits in (1, 2, 3, 5):
            values = list(range(1 << bits)) * 3
            wire = int.from_bytes(pack(values, bits), "little")
            self.assertEqual([(wire >> (i * bits)) & ((1 << bits) - 1) for i in range(len(values))], values)

    def test_exact_payload_lengths_and_determinism(self):
        for bits in (2, 3):
            for n, k in ((1, 32), (7, 96), (128, 256)):
                blob, dense = fixture(bits, n, k)
                self.assertEqual(len(blob), 24 + n * k * bits // 8 + (n * k // 32 + 7) // 8 + n * 7)
                self.assertEqual(struct.unpack("<4sBBHQQ", blob[:24]), (f"SQ{bits}\0".encode(), 1, 120, 0, n, k))
                self.assertEqual(len(dense), n * k * 4)
                self.assertEqual((blob, dense), fixture(bits, n, k))


if __name__ == "__main__":
    unittest.main()
