# test_spacy_gpu_ner.py
"""
Run this from your venv to verify:
  1) PyTorch CUDA works + which GPU is visible
  2) spaCy is actually configured to use GPU backend (Thinc ops)
  3) spaCy transformer is running and allocating CUDA memory
  4) End-to-end NER call works (with your project NER module if available)

Usage (from repo root):
  .venv\Scripts\python.exe test_spacy_gpu_ner.py
"""

from __future__ import annotations

import os
import sys
import time


def _hr(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_torch_cuda() -> bool:
    _hr("1) PyTorch CUDA check")

    try:
        import torch
    except Exception as e:
        print("FAIL: could not import torch:", repr(e))
        return False

    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda mem allocated (start):", torch.cuda.memory_allocated())
        # allocate a tiny tensor to force CUDA context init
        x = torch.randn(1024, device="cuda")
        torch.cuda.synchronize()
        print("cuda mem allocated (after tiny alloc):", torch.cuda.memory_allocated())
        del x
        torch.cuda.synchronize()
        return True

    return False


def test_spacy_backend_and_ops(force_gpu: bool = True):
    _hr("2) spaCy + Thinc backend check (ops should be CupyOps if GPU-backed)")

    try:
        import spacy
    except Exception as e:
        print("FAIL: could not import spacy:", repr(e))
        return None

    print("spacy:", spacy.__version__)

    if force_gpu:
        try:
            spacy.require_gpu()
            print("spacy.require_gpu(): OK (GPU required)")
        except Exception as e:
            print("spacy.require_gpu(): FAILED:", repr(e))

    try:
        nlp = spacy.load("en_core_web_trf")
        print("Loaded model: en_core_web_trf")
    except Exception as e:
        print("FAIL: could not load en_core_web_trf:", repr(e))
        return None

    print("nlp.pipe_names:", nlp.pipe_names)

    try:
        trf = nlp.get_pipe("transformer")
        ops_type = type(trf.model.ops)
        ops_name = getattr(trf.model.ops, "name", None)
        print("transformer ops type:", ops_type)
        print("transformer ops name:", ops_name)
        # expected:
        #  - GPU: CupyOps
        #  - CPU: NumpyOps
    except Exception as e:
        print("WARN: could not inspect transformer ops:", repr(e))

    return nlp


def test_spacy_cuda_memory(nlp):
    _hr("3) Does spaCy transformer allocate CUDA memory during a forward pass?")

    try:
        import torch
    except Exception as e:
        print("SKIP: torch import failed:", repr(e))
        return

    if not torch.cuda.is_available():
        print("SKIP: torch.cuda.is_available() is False")
        return

    before = torch.cuda.memory_allocated()
    print("cuda mem allocated (before):", before)

    texts = [
        "Ralph Wilson scored for Track & Field. Colin Vincent Foster attended.",
        "JakeSmith met MacArthur and OConnor at the game.",
        "A T H L E T I C S awards were given to John Doe.",
    ]

    # warmup pass (first call often includes extra overhead)
    _ = list(nlp.pipe(texts, batch_size=8, n_process=1))
    torch.cuda.synchronize()

    mid = torch.cuda.memory_allocated()
    print("cuda mem allocated (after warmup):", mid)

    # timed pass
    t0 = time.perf_counter()
    docs = list(nlp.pipe(texts * 50, batch_size=32, n_process=1))
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    after = torch.cuda.memory_allocated()
    print("cuda mem allocated (after timed run):", after)
    print("timed run docs:", len(docs), "seconds:", round(t1 - t0, 3))

    # quick sanity: show entities from first doc
    print("\nSample ents from first doc:")
    for ent in docs[0].ents:
        if ent.label_ == "PERSON":
            print("  PERSON:", ent.text)

    if after <= before:
        print("\nNOTE: CUDA memory did not increase. This often means the model ran on CPU backend.")
    else:
        print("\nOK: CUDA memory increased during spaCy run (strong evidence of GPU execution).")


def test_project_ner_module():
    _hr("4) Project module check: src.name_ner (if present)")

    try:
        from src.name_ner import extract_person_names_from_candidates, extract_person_names_per_candidate
    except Exception as e:
        print("SKIP: could not import src.name_ner:", repr(e))
        return

    candidates = [
        "Ralph Wilson",
        "Wilson",
        "Colin Vincent Foster",
        "Colin Vincent",
        "Vincent Foster",
        "JakeSmith",
        "MacArthur",
        "OConnor",
        "A T H L E T I C S",  # should typically not become a PERSON
    ]

    t0 = time.perf_counter()
    per = extract_person_names_per_candidate(candidates, verbose=True)
    t1 = time.perf_counter()

    flat = []
    for lst in per:
        flat.extend(lst)

    t2 = time.perf_counter()
    deduped = extract_person_names_from_candidates(candidates, verbose=False)
    t3 = time.perf_counter()

    print("\nPer-candidate PERSON outputs:")
    for c, lst in zip(candidates, per):
        print(f"  {c!r} -> {lst}")

    print("\nFlattened:", flat)
    print("Deduped (nested removed):", deduped)

    print("\nTiming:")
    print("  per-candidate call:", round(t1 - t0, 3), "sec")
    print("  deduped wrapper:", round(t3 - t2, 3), "sec")

    print("\nExpected behavior:")
    print("  - If both 'Ralph Wilson' and 'Wilson' are extracted, dedupe should drop 'Wilson'.")
    print("  - If 'Colin Vincent' and 'Vincent Foster' overlap from 'Colin Vincent Foster', overlap merge should output the full name.")


def main():
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("CWD:", os.getcwd())

    torch_ok = test_torch_cuda()

    nlp = test_spacy_backend_and_ops(force_gpu=True)
    if nlp is not None and torch_ok:
        test_spacy_cuda_memory(nlp)

    test_project_ner_module()

    _hr("Done")


if __name__ == "__main__":
    main()
