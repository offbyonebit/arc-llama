from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_llama import model_compatibility as compat
from arc_llama.config import Config, ModelConfig


def header(architecture: str) -> bytes:
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    return b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + string('general.architecture') + struct.pack('<I', 8) + string(architecture)


@pytest.mark.parametrize('data', [b'', b'GGUF', header('llama')[:-2], header('../llama'), b'NOT!' + header('llama')[4:]])
def test_unusable_metadata_stays_unknown(data):
    assert compat.header_architecture(data) is None


def test_architecture_comes_from_metadata_not_filename():
    assert compat.header_architecture(header('lfm2')) == 'lfm2'


@pytest.mark.parametrize('body', [{'repo':'../bad','file':'m.gguf'}, {'repo':'u/r','file':'../m.gguf'}, {'repo':'u/r','file':'/m.gguf'}, {'repo':'u/r','file':'m\\x.gguf'}, {'repo':'u/r','file':'m.txt'}, {'name':42}])
def test_untrusted_request_paths_rejected(body):
    with pytest.raises(ValueError):
        compat.validate_request(body)


def test_runtime_identity_changes_with_adjacent_library(tmp_path):
    runtime = tmp_path/'llama-server'
    runtime.write_bytes(b'fake')
    library = tmp_path/'libllama.so'
    library.write_bytes(b'one')
    first = compat.runtime_identity(str(runtime))
    library.write_bytes(b'two-updated')
    assert first != compat.runtime_identity(str(runtime))
    runtime.unlink()
    assert compat.runtime_identity(str(runtime)) is None


@pytest.mark.parametrize(('diagnostic','expected'), [
    ("unknown model architecture: 'newarch'",'rejected'),
    ('key not found in model: newarch.context_length','recognized'),
    ('unknown model architecture: otherarch','unknown'),
    ('device not available','unknown'),
    ('key not found in model: tokenizer.ggml.tokens','unknown'),
])
def test_probe_requires_architecture_specific_evidence(monkeypatch,diagnostic,expected):
    compat._PROBE_CACHE.clear()
    def run(argv, **kwargs):
        assert '--device' in argv and argv[argv.index('--device')+1]=='none'
        assert compat.header_architecture(Path(argv[2]).read_bytes()) == 'newarch'
        assert kwargs['timeout']==10
        return SimpleNamespace(returncode=1,stdout=b'',stderr=diagnostic.encode())
    monkeypatch.setattr(compat.subprocess,'run',run)
    assert compat.probe_architecture('runtime','identity','newarch')==expected
    compat._PROBE_CACHE.clear()


def test_remote_assessment_pins_selected_file_and_does_not_claim_inference(monkeypatch):
    import huggingface_hub
    cfg=Config()
    cfg.paths.llama_server='test-runtime'
    sha='a'*40
    monkeypatch.setattr(compat,'runtime_identity',lambda _:('/runtime','fingerprint'))
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:SimpleNamespace(model_info=lambda _:SimpleNamespace(sha=sha,siblings=[SimpleNamespace(rfilename='m.gguf')])))
    reads=[]
    monkeypatch.setattr(compat,'remote_header',lambda repo,file,revision:reads.append((repo,file,revision)) or header('llama'))
    monkeypatch.setattr(compat,'probe_architecture',lambda *args:'recognized')
    result=compat.assess_compatibility(cfg,{'repo':'u/r','file':'m.gguf'})
    assert reads==[('u/r','m.gguf',sha)]
    assert result['status']=='recognized' and 'unverified' in result['scope']
    assert result['revision']==sha
    with pytest.raises(ValueError):
        compat.assess_compatibility(cfg,{'repo':'u/r','file':'wrong.gguf'})
    assert len(reads)==1


def test_local_rejection_and_changed_runtime_remain_distinct(tmp_path,monkeypatch):
    cfg=Config()
    file=tmp_path/'looks-like-llama.gguf'
    file.write_bytes(header('newarch'))
    cfg.models=[ModelConfig('local',str(file),18080,'gpu')]
    monkeypatch.setattr(compat,'runtime_identity',lambda _:('/runtime','old'))
    monkeypatch.setattr(compat,'probe_architecture',lambda *args:'rejected')
    result=compat.assess_compatibility(cfg,{'name':'local'})
    assert result['status']=='incompatible' and result['architecture']=='newarch'
    identities=iter([('/runtime','old'),('/runtime','new')])
    monkeypatch.setattr(compat,'runtime_identity',lambda _:next(identities))
    assert compat.assess_compatibility(cfg,{'name':'local'})['status']=='unknown'


async def test_compatibility_endpoint_auth_and_validation(monkeypatch,tmp_path):
    from httpx import ASGITransport, AsyncClient
    from test_model_library import _app
    _,app=_app(monkeypatch,tmp_path)
    calls=[]
    monkeypatch.setattr(compat,'assess_compatibility',lambda cfg,body:calls.append(body) or {'status':'unknown'})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app),base_url='http://t') as client:
            assert (await client.post('/admin/library/compatibility',json={'repo':'u/r','file':'m.gguf'})).status_code==401
            auth={'Authorization':'Bearer tok'}
            assert (await client.post('/admin/library/compatibility',json={'repo':'u/r','file':'../m.gguf'},headers=auth)).status_code==400
            response=await client.post('/admin/library/compatibility',json={'repo':'u/r','file':'m.gguf'},headers=auth)
            assert response.json()=={'status':'unknown'}
    assert calls==[{'repo':'u/r','file':'m.gguf'}]


def test_remote_read_is_bounded_even_when_range_is_ignored(monkeypatch):
    import httpx
    blocks = []
    closed = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for index in range(100):
                blocks.append(index)
                yield b'x' * 16384
        def close(self):
            closed.append(True)
    from contextlib import contextmanager

    @contextmanager
    def request(method, url, **kwargs):
        assert method == 'GET'
        assert '/'+('a'*40)+'/' in url
        assert kwargs['headers']['Range']=='bytes=0-262143'
        response = httpx.Response(200, stream=Stream(), request=httpx.Request(method, url))
        try:
            yield response
        finally:
            response.close()
    monkeypatch.setattr(compat.httpx,'stream',request)
    assert len(compat.remote_header('u/r','m.gguf','a'*40))==262144
    assert len(blocks)==16 and closed


def test_unknown_probe_is_retryable(monkeypatch):
    compat._PROBE_CACHE.clear()
    calls=[]
    def run(*args, **kwargs):
        calls.append(True)
        if len(calls)==1:
            raise compat.subprocess.TimeoutExpired(args[0],10)
        return SimpleNamespace(returncode=1,stdout=b'',stderr=b'key not found in model: llama.context_length')
    monkeypatch.setattr(compat.subprocess,'run',run)
    assert compat.probe_architecture('runtime','identity','llama')=='unknown'
    assert compat.probe_architecture('runtime','identity','llama')=='recognized'
    assert compat.probe_architecture('runtime','identity','llama')=='recognized'
    assert len(calls)==2
    compat._PROBE_CACHE.clear()


def test_registered_file_identity_changes_when_file_is_replaced(tmp_path):
    from arc_llama.model_library import file_readiness
    path=tmp_path/'m.gguf'
    path.write_bytes(header('llama'))
    model=ModelConfig('local',str(path),18080,'gpu')
    before=file_readiness(model)['identity']
    path.write_bytes(header('qwen3'))
    assert before != file_readiness(model)['identity']
