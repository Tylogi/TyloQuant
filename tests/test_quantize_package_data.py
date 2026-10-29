"""Exercise a real wheel: source-tree availability cannot prove installation."""
import os
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def test_built_wheel_contains_all_offline_cuda_sources_and_alphaq_recipes(tmp_path):
    if int(version('setuptools').split('.')[0]) < 61:
        pytest.skip('wheel build needs setuptools>=61, as declared in pyproject.toml')
    source = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(source),
        '--no-deps', '--no-build-isolation', '--no-index', '--wheel-dir', str(tmp_path/'dist')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((tmp_path/'dist').glob('mfq-*.whl'))
    assert len(wheels) == 1, result.stdout + result.stderr
    wheel = wheels[0]
    installed = tmp_path/'installed'
    with zipfile.ZipFile(wheel) as package:
        package.extractall(installed)
    program = """
from pathlib import Path
import mfq
from mfq.quantize.cuda import _ext
from mfq.calibration.alphaq_workflow import automatic_recipe, recipe_types
root = Path.cwd()
assert Path(mfq.__file__).resolve().is_relative_to(root)
assert len(_ext._SOURCES) == 9
for path in _ext._SOURCES:
    assert Path(path).is_file(), path
assert (_ext._DIR/'nvq_chunk.h').is_file()
for relative in ['kernels/hip/tq.py', 'kernels/hip/tq_ops.hip',
                 'kernels/hip/mfq_hip_compat.h', 'quantize/hip/tq_encoder.hip',
                 'quantize/hip/tq_fitter.hip']:
    assert (Path(mfq.__file__).parent/relative).is_file(), relative
recipes = list((Path(mfq.__file__).parent/'calibration/recipes').glob('*.json'))
assert len(recipes) == 9
for recipe in recipes:
    assert recipe_types(recipe)
nemotron = sorted((Path(mfq.__file__).parent/'calibration/recipes/nemotron_h').glob('*.json'))
assert [path.stem for path in nemotron] == ['IQ2_M', 'IQ4_XS']
for recipe in nemotron:
    assert recipe_types(recipe)
config = dict(model_type='qwen4_exp_text', hidden_size=2560, num_hidden_layers=48,
              num_experts=512, num_experts_per_tok=10, vocab_size=248320)
assert automatic_recipe(config, 3.56583659).stem == 'Q2_K_XL'
"""
    environment = {**os.environ, 'PYTHONPATH': str(installed), 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-c', program], cwd=installed,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
