"""Overlap DataLoader + CPU/GPU batch prep with GPU training steps."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class TowerPrepStreams:
    """Persistent CUDA streams reused across training steps (avoid per-step alloc)."""

    prep: torch.cuda.Stream
    vlm: torch.cuda.Stream
    vggt: torch.cuda.Stream

    @classmethod
    def for_device(cls, device: torch.device) -> "TowerPrepStreams":
        if device.type != "cuda":
            raise ValueError("TowerPrepStreams requires a CUDA device.")
        return cls(
            prep=torch.cuda.Stream(device=device),
            vlm=torch.cuda.Stream(device=device),
            vggt=torch.cuda.Stream(device=device),
        )


class TrainingBatchIterator:
    """Reuse CPU/GPU prefetch pipelines across epoch boundaries."""

    def __init__(
        self,
        loader: DataLoader,
        *,
        sampler: Any,
        prepare_cpu: Callable[..., Dict[str, Any]],
        prepare_gpu: Callable[[Dict[str, Any]], Dict[str, Any]],
        device: torch.device,
        cpu_prefetch: int = 0,
        gpu_pipeline: bool = False,
        prep_stream: Optional[torch.cuda.Stream] = None,
        gpu_pipeline_depth: int = 2,
    ) -> None:
        self.loader = loader
        self.sampler = sampler
        self.prepare_cpu = prepare_cpu
        self.prepare_gpu = prepare_gpu
        self.device = device
        self.cpu_prefetch = cpu_prefetch
        self.gpu_pipeline = gpu_pipeline
        self.prep_stream = prep_stream
        self.gpu_pipeline_depth = gpu_pipeline_depth
        self.epoch = 0

    def _start_epoch(self) -> Iterator[Dict[str, Any]]:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)
        return iter_training_batches(
            self.loader,
            prepare_cpu=self.prepare_cpu,
            prepare_gpu=self.prepare_gpu,
            device=self.device,
            cpu_prefetch=self.cpu_prefetch,
            gpu_pipeline=self.gpu_pipeline,
            prep_stream=self.prep_stream,
            gpu_pipeline_depth=self.gpu_pipeline_depth,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            yield from self._start_epoch()
            self.epoch += 1


class CpuBatchPrefetcher:
    """Prefetch CPU-side ``prepare_model_batch_cpu`` results in a background thread."""

    def __init__(
        self,
        loader: DataLoader,
        prepare_cpu: Callable[..., Dict[str, Any]],
        *,
        prepare_kwargs: Optional[Dict[str, Any]] = None,
        max_prefetch: int = 2,
    ) -> None:
        self.loader = loader
        self.prepare_cpu = prepare_cpu
        self.prepare_kwargs = dict(prepare_kwargs or {})
        self.max_prefetch = max(1, int(max_prefetch))
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.max_prefetch)
        self._sentinel = object()
        self._thread: Optional[threading.Thread] = None
        self._exc: Optional[BaseException] = None

    def _worker(self) -> None:
        try:
            for batch in self.loader:
                self._queue.put(self.prepare_cpu(batch, **self.prepare_kwargs))
            self._queue.put(self._sentinel)
        except BaseException as exc:  # pragma: no cover - surfaced on main thread
            self._exc = exc
            self._queue.put(self._sentinel)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                if self._exc is not None:
                    raise self._exc
                break
            yield item


class AsyncGpuPrepPipeline:
    """Background GPU tower prep queue; overlaps VLM/VGGT with action backward."""

    def __init__(
        self,
        cpu_batches: Iterator[Dict[str, Any]],
        prepare_gpu: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        device: torch.device,
        prep_stream: Optional[torch.cuda.Stream] = None,
        queue_depth: int = 2,
    ) -> None:
        self._cpu_batches = cpu_batches
        self._prepare_gpu = prepare_gpu
        self._device = device
        self._queue_depth = max(1, int(queue_depth))
        self._prep_stream = prep_stream
        if self._prep_stream is None and device.type == "cuda":
            self._prep_stream = torch.cuda.Stream(device=device)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._queue_depth)
        self._sentinel = object()
        self._thread: Optional[threading.Thread] = None
        self._exc: Optional[BaseException] = None

    def _worker(self) -> None:
        if self._device.type == "cuda":
            idx = self._device.index if self._device.index is not None else 0
            torch.cuda.set_device(idx)
        try:
            for cpu_payload in self._cpu_batches:
                if self._prep_stream is None:
                    ready = self._prepare_gpu(cpu_payload)
                    self._queue.put(ready)
                    continue
                with torch.cuda.stream(self._prep_stream):
                    ready = self._prepare_gpu(cpu_payload)
                done = torch.cuda.Event()
                done.record(self._prep_stream)
                self._queue.put((ready, done))
            self._queue.put(self._sentinel)
        except BaseException as exc:  # pragma: no cover - surfaced on main thread
            self._exc = exc
            self._queue.put(self._sentinel)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        return self

    def __next__(self) -> Dict[str, Any]:
        item = self._queue.get()
        if item is self._sentinel:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        if self._prep_stream is None:
            return item
        ready, done = item
        done.synchronize()
        return ready


class GpuBatchPipeline:
    """Double-buffer GPU tower prep on a side stream while the train stream runs."""

    def __init__(
        self,
        cpu_batches: Iterator[Dict[str, Any]],
        prepare_gpu: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        device: torch.device,
        prep_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self._cpu_batches = cpu_batches
        self._prepare_gpu = prepare_gpu
        self._device = device
        self._prep_stream = prep_stream
        if self._prep_stream is None and device.type == "cuda":
            self._prep_stream = torch.cuda.Stream(device=device)
        self._next_batch: Optional[Dict[str, Any]] = None

    def _launch_gpu_prep(self, cpu_payload: Dict[str, Any]) -> None:
        if self._prep_stream is None:
            self._next_batch = self._prepare_gpu(cpu_payload)
            return
        self._prep_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._prep_stream):
            self._next_batch = self._prepare_gpu(cpu_payload)

    def _wait_ready(self) -> Dict[str, Any]:
        if self._prep_stream is not None:
            torch.cuda.current_stream().wait_stream(self._prep_stream)
        if self._next_batch is None:
            raise RuntimeError("GpuBatchPipeline has no prepared batch.")
        return self._next_batch

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        cpu_iter = iter(self._cpu_batches)
        try:
            first_cpu = next(cpu_iter)
        except StopIteration:
            return
        self._launch_gpu_prep(first_cpu)
        for cpu_payload in cpu_iter:
            ready = self._wait_ready()
            self._launch_gpu_prep(cpu_payload)
            yield ready
        yield self._wait_ready()


def iter_training_batches(
    loader: DataLoader,
    *,
    prepare_cpu: Callable[..., Dict[str, Any]],
    prepare_gpu: Callable[[Dict[str, Any]], Dict[str, Any]],
    device: torch.device,
    cpu_prefetch: int = 0,
    gpu_pipeline: bool = False,
    prep_stream: Optional[torch.cuda.Stream] = None,
    gpu_pipeline_depth: int = 2,
) -> Iterator[Dict[str, Any]]:
    """Build a batch iterator with optional CPU and GPU prefetch/pipeline stages."""
    batch_source: Iterator[Dict[str, Any]]
    if cpu_prefetch > 0:
        batch_source = CpuBatchPrefetcher(
            loader,
            prepare_cpu,
            max_prefetch=cpu_prefetch,
        )
    else:
        batch_source = (prepare_cpu(batch) for batch in loader)

    if gpu_pipeline and device.type == "cuda":
        # GPU prep must stay on the training thread: VLM/VGGT modules are not
        # safe to forward concurrently with action_expert backward.
        return iter(
            GpuBatchPipeline(
                batch_source,
                prepare_gpu,
                device=device,
                prep_stream=prep_stream,
            )
        )
    return (prepare_gpu(cpu_payload) for cpu_payload in batch_source)
