import json

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from mfq.calibration.chat import CalibrationChatRenderer, prepare_calibration_corpus
from mfq.calibration.dataset import _FORMAT, _write_corpus_artifact, build_corpus_from_records


class TokenizerFixture:
    bos_token = '<bos>'
    eos_token = '<eos>'
    eos_token_id = 2
    pad_token = '<pad>'
    unk_token = '<unk>'
    all_special_tokens = ['<bos>', '<eos>', '<turn>', '<end>']
    chat_template = 'fixture'
    name_or_path = 'fixture'
    vocab_size = 1024

    def encode(self, text, **kwargs):
        assert not kwargs.get('add_special_tokens', False)
        return [ord(char) + 10 for char in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(token - 10) if token >= 10 else '<eos>' for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = '<bos>' + ''.join('<turn>' + message['role'] + '\n' + message['content'] + '<end>' for message in messages)
        if add_generation_prompt:
            text += '<turn>assistant\n'
        return self.encode(text) if tokenize else text


@pytest.mark.parametrize('text', ['ordinary text', '汉字文本', 'a paragraph mentioning <turn> midway'])
def test_auto_wraps_unformatted_records_using_model_template(text):
    tokenizer = TokenizerFixture()
    renderer = CalibrationChatRenderer(tokenizer)
    expected = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
    assert renderer.encode(text) == expected + [2]
    assert renderer.counts['wrapped'] == 1


@pytest.mark.parametrize('bos,role', [('<bos>', 'user'), ('', 'user'), ('<bos>', 'system'), ('', 'assistant')])
def test_preformatted_records_are_not_wrapped_twice(bos, role):
    tokenizer = TokenizerFixture()
    text = bos + '<turn>' + role + '\nHello<end>'
    renderer = CalibrationChatRenderer(tokenizer)
    assert renderer.encode(text) == tokenizer.encode(text) + [2]
    assert renderer.counts == dict(wrapped=0, preformatted=1, plain=0, conversations=0)


def test_template_opt_out_and_base_model_keep_plain_text():
    tokenizer = TokenizerFixture()
    assert CalibrationChatRenderer(tokenizer, enabled=False).encode('hello') == tokenizer.encode('hello') + [2]
    tokenizer.chat_template = None
    renderer = CalibrationChatRenderer(tokenizer)
    assert renderer.encode('hello') == tokenizer.encode('hello') + [2]
    assert renderer.summary()['model_has_template'] is False


def test_huggingface_template_and_named_template_dictionary():
    raw = Tokenizer(WordLevel({'[UNK]': 0, '[BOS]': 1, '[EOS]': 2, '[USER]': 3, '[ASSISTANT]': 4, '[END]': 5, 'hello': 6}, unk_token='[UNK]'))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]',
        additional_special_tokens=['[USER]', '[ASSISTANT]', '[END]'])
    tokenizer.chat_template = {'default': "{{ bos_token }} {% for m in messages %}{{ '[USER]' if m.role == 'user' else '[ASSISTANT]' }} {{ m.content }} [END] {% endfor %}{% if add_generation_prompt %}[ASSISTANT]{% endif %}"}
    renderer = CalibrationChatRenderer(tokenizer)
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'hello'}], tokenize=False, add_generation_prompt=True)
    expected = tokenizer.apply_chat_template([{'role': 'user', 'content': 'hello'}], tokenize=True, add_generation_prompt=True)
    expected = list(expected['input_ids']) + [2]
    assert renderer.encode('hello') == expected
    assert renderer.encode(text) == expected


@pytest.mark.parametrize('extension', ['.txt', '.json', '.jsonl'])
def test_raw_file_becomes_a_reusable_corpus_with_rendering_metadata(tmp_path, extension):
    path = tmp_path / ('input' + extension)
    if extension == '.txt':
        path.write_text('first record\n\n<turn>user\nsecond<end>')
    else:
        records = [{'text': 'first record'}, {'messages': [{'role': 'user', 'content': 'second'}, {'role': 'assistant', 'content': 'answer'}]}]
        path.write_text(json.dumps(records) if extension == '.json' else '\n'.join(json.dumps(record) for record in records))
    original = path.read_bytes()
    with prepare_calibration_corpus(path, 'unused', tokenizer=TokenizerFixture(), window_length=1024) as corpus:
        root = corpus.root
        assert root.is_dir()
        assert corpus.token_count('train') > 0
        assert corpus.token_count('validation') == 0
        assert corpus.manifest['chat_rendering']['wrapped'] == 1
        if extension == '.txt':
            assert corpus.manifest['chat_rendering']['preformatted'] == 1
        else:
            assert corpus.manifest['chat_rendering']['conversations'] == 1
            assert 'answer' in TokenizerFixture().decode(corpus.chunk_tokens(1))
    assert not root.exists()
    assert path.read_bytes() == original


def test_prepared_plain_corpus_wraps_without_retokenizing_body_or_changing_splits(tmp_path):
    tokenizer = TokenizerFixture()
    source = tmp_path / 'saved'
    body = tokenizer.encode('original body') + [2]
    manifest = {'format': _FORMAT, 'domains': ['test'], 'tokenizer': {'render_mode': 'plain'},
        'actual_tokens': {'train': len(body), 'validation': len(body)}}
    _write_corpus_artifact(source, [np.array(body), np.array(body)], [0, 1], [0, 0], manifest).close()
    original = (source / 'tokens.npy').read_bytes()
    with prepare_calibration_corpus(source, 'unused', tokenizer=tokenizer, window_length=1024) as corpus:
        assert corpus.split_ids.tolist() == [0, 1]
        rendered = corpus.chunk_tokens(0).tolist()
        prefix = tokenizer.encode('<bos><turn>user\n')
        assert rendered[len(prefix):len(prefix) + len(body) - 1] == body[:-1]
        assert corpus.manifest['chat_rendering']['wrapped'] == 2
    assert (source / 'tokens.npy').read_bytes() == original
    with prepare_calibration_corpus(source, 'unused', apply_chat_template=False) as corpus:
        assert corpus.root == source
        assert corpus.chunk_tokens(0).tolist() == body


def test_prepared_chat_corpus_is_reused_without_loading_tokenizer(tmp_path):
    source = tmp_path / 'saved'
    manifest = {'format': _FORMAT, 'domains': ['test'], 'tokenizer': {'render_mode': 'trace-chat'}, 'actual_tokens': {'train': 3, 'validation': 0}}
    _write_corpus_artifact(source, [np.array([17, 18, 2])], [0], [0], manifest).close()
    with prepare_calibration_corpus(source, 'tokenizer-does-not-exist') as corpus:
        assert corpus.root == source
        assert corpus.chunk_tokens(0).tolist() == [17, 18, 2]


def test_corpus_builder_defaults_to_auto_and_preserves_disjoint_records(tmp_path):
    corpus = build_corpus_from_records({'text': [f'record {index}' for index in range(10)]}, TokenizerFixture(), tmp_path / 'auto',
        train_tokens=80, validation_tokens=80, sequence_length=80)
    try:
        assert corpus.manifest['tokenizer']['render_mode'] == 'auto'
        assert corpus.manifest['chat_rendering']['wrapped'] > 0
        assert corpus.token_count('train') == corpus.token_count('validation') == 80
    finally:
        corpus.close()


def test_plain_raw_file_opt_out_and_invalid_messages(tmp_path):
    source = tmp_path / 'records.jsonl'
    source.write_text(json.dumps({'text': 'hello'}))
    with prepare_calibration_corpus(source, 'unused', tokenizer=TokenizerFixture(), apply_chat_template=False) as corpus:
        assert corpus.chunk_tokens(0).tolist() == TokenizerFixture().encode('hello') + [2]
        assert corpus.manifest['chat_rendering']['enabled'] is False
    source.write_text(json.dumps({'messages': [{'role': 'invalid', 'content': 'hello'}]}))
    with pytest.raises(ValueError, match='line 1'), prepare_calibration_corpus(source, 'unused', tokenizer=TokenizerFixture()):
        pytest.fail('invalid conversation accepted')


def test_cli_defaults_to_auto_and_accepts_explicit_opt_out():
    from mfq.cli import _build_parser
    arguments = ['calibrate', 'imatrix', '--model', 'model', '--corpus', 'data.txt', '--output', 'out.imatrix']
    assert _build_parser().parse_args(arguments).render_mode == 'auto'
    assert _build_parser().parse_args([*arguments, '--render-mode', 'plain']).render_mode == 'plain'


@pytest.mark.parametrize('mode', ['auto', 'plain'])
def test_cli_prepares_raw_records_before_imatrix_collection(tmp_path, monkeypatch, mode):
    import transformers

    import mfq.calibration.imatrix as imatrix_module
    from mfq.cli import _build_parser, _calibrate_imatrix
    path = tmp_path / 'input.txt'
    path.write_text('hello world')
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}), encoding='utf-8')
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: TokenizerFixture())
    seen = []
    def collect(model, corpus, output, **kwargs):
        seen.append(corpus.manifest['chat_rendering'])
        text = TokenizerFixture().decode(corpus.chunk_tokens(0))
        assert ('<turn>user' in text) == (mode == 'auto')
        assert 'hello world' in text
    monkeypatch.setattr(imatrix_module, 'collect_imatrix', collect)
    args = _build_parser().parse_args(['calibrate', 'imatrix', '--model', str(tmp_path), '--corpus', str(path),
        '--output', str(tmp_path / 'out.imatrix'), '--backend', 'metal', '--render-mode', mode])
    assert _calibrate_imatrix(args) == 0
    assert seen[0]['enabled'] == (mode == 'auto')
