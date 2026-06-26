"""Unit tests for scripts/run_inference_worker.work_loop (plan Phase 1).

GPU-free: drives the queue-drain loop with the real ClaimStore over an in-memory
fake bucket and a stub `process_shard`, so the claim->process->done ordering,
resume, and multi-worker exactly-once behaviour are exercised without torch/GCS.
"""

from __future__ import annotations

import sys
from pathlib import Path

from google.api_core.exceptions import NotFound, PreconditionFailed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from inference.claim import ClaimStore  # noqa: E402
from run_inference_worker import work_loop  # noqa: E402


class _FakeBlob:
    def __init__(self, store, name):
        self._store, self.name = store, name

    def upload_from_string(self, data, if_generation_match=None):
        if if_generation_match == 0 and self.name in self._store:
            raise PreconditionFailed(self.name)
        gen = self._store.get(self.name, (0, None))[0] + 1
        self._store[self.name] = (gen, data if isinstance(data, bytes) else data.encode())

    def download_as_text(self):
        if self.name not in self._store:
            raise NotFound(self.name)
        return self._store[self.name][1].decode()

    def delete(self, if_generation_match=None):
        if self.name not in self._store:
            raise NotFound(self.name)
        del self._store[self.name]


class _FakeBucket:
    def __init__(self):
        self._store = {}

    def blob(self, name):
        return _FakeBlob(self._store, name)

    def list_blobs(self, prefix=""):
        return [_FakeBlob(self._store, n) for n in sorted(self._store) if n.startswith(prefix)]


SHARDS = [f"shard_{i:06d}" for i in range(8)]


def test_single_worker_drains_all_exactly_once():
    bucket = _FakeBucket()
    store = ClaimStore(bucket, "inf/run", worker_id="A")
    processed = []
    n = work_loop(store, SHARDS, processed.append)
    assert n == 8
    assert processed == SHARDS                 # in order, each once
    assert store.done_ids() == set(SHARDS)     # all marked done


def test_mark_done_only_after_process():
    bucket = _FakeBucket()
    store = ClaimStore(bucket, "inf/run", worker_id="A")
    order = []
    def proc(sid):
        # at process time the shard is claimed but NOT yet done
        order.append((sid, sid in store.done_ids()))
    work_loop(store, SHARDS, proc)
    assert all(done is False for _, done in order)


def test_resume_skips_already_done():
    bucket = _FakeBucket()
    store = ClaimStore(bucket, "inf/run", worker_id="A")
    for sid in SHARDS[:5]:
        store.mark_done(sid)
    processed = []
    n = work_loop(store, SHARDS, processed.append)
    assert n == 3
    assert processed == SHARDS[5:]


def test_two_workers_cover_all_disjointly():
    """Worker A does a few, B drains the rest: union == all, no overlap."""
    bucket = _FakeBucket()
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    b = ClaimStore(bucket, "inf/run", worker_id="B")
    proc_a, proc_b = [], []
    work_loop(a, SHARDS, proc_a.append, max_shards=3)
    work_loop(b, SHARDS, proc_b.append)
    assert len(proc_a) == 3
    assert set(proc_a) & set(proc_b) == set()        # disjoint
    assert set(proc_a) | set(proc_b) == set(SHARDS)   # complete
    assert a.done_ids() == set(SHARDS)


def test_max_shards_stops_early():
    bucket = _FakeBucket()
    store = ClaimStore(bucket, "inf/run", worker_id="A")
    processed = []
    n = work_loop(store, SHARDS, processed.append, max_shards=2)
    assert n == 2
    assert len(processed) == 2
    assert store.done_ids() == set(SHARDS[:2])
