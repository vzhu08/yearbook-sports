#!/usr/bin/env python3
"""
Predict-only PaddleOCR service + client.

- Client: run_paddle_ocr(images, device="gpu", batch_size=8, save_json_dir=..., save_img_dir=...)
- Service: loads PaddleOCR once, runs predict on a list of image paths,
  then (after inference) saves JSON/overlays via res.save_to_json/save_to_img.

We keep stdout strictly JSON so the caller never sees garbage.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import tempfile
import subprocess
from typing import Any, Dict, List

# -----------------------
# Client (safe to import)
# -----------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _parse_stdout(stdout: str) -> Dict[str, Any]:
    s = _ANSI_RE.sub("", stdout or "").strip()
    if not s:
        return {"results": {}, "errors": {"_parse": "empty stdout"}}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
        return {"results": {}, "errors": {"_parse": "Failed to parse service JSON", "_raw": s[:1000]}}


def run_paddle_ocr(
    images: List[str],
    *,
    device: str = "gpu",
    lang: str = "en",
    batch_size: int | None = 8,
    python_exe: str = sys.executable,
    service_script: str | None = None,
    save_json_dir: str | None = None,
    save_img_dir: str | None = None,
) -> Dict[str, Any]:
    images = [p for p in images if p and os.path.exists(p)]
    if not images:
        return {"results": {}, "errors": {"_client": "No valid images"}}

    if service_script is None:
        service_script = os.path.abspath(__file__)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump({"images": images}, tf, ensure_ascii=False)
        req_path = tf.name

    try:
        cmd = [python_exe, service_script, "--device", device, "--lang", lang, "--json", req_path]
        if batch_size is not None and batch_size > 0:
            cmd += ["--batch_size", str(batch_size)]
        if save_json_dir:
            os.makedirs(save_json_dir, exist_ok=True)
            cmd += ["--save_json_dir", save_json_dir]
        if save_img_dir:
            os.makedirs(save_img_dir, exist_ok=True)
            cmd += ["--save_img_dir", save_img_dir]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        out = _parse_stdout(proc.stdout)
        if proc.returncode != 0:
            out.setdefault("errors", {})["_proc"] = f"exit {proc.returncode}"
            if proc.stderr:
                out["errors"]["_stderr"] = proc.stderr[-1000:]
        return out
    finally:
        try:
            os.unlink(req_path)
        except Exception:
            pass


# -----------------------
# Service (run as script)
# -----------------------
def _prepend_paddle_nv_bins() -> None:
    """(Windows) prioritize venv’s packaged CUDA/cuDNN DLLs if present."""
    try:
        import site, glob
        sites = list(getattr(site, "getsitepackages", lambda: [])()) + sys.path
        sp = next((p for p in sites if p.endswith("site-packages") and os.path.isdir(p)), None)
        if not sp:
            return
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except Exception:
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _service_main() -> None:
    os.environ.setdefault("FLAGS_minloglevel", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    _prepend_paddle_nv_bins()

    payload: Dict[str, Any] = {"results": {}, "errors": {}}

    try:
        from paddleocr import PaddleOCR  # type: ignore

        ap = argparse.ArgumentParser(description="PaddleOCR JSON CLI (predict-only)")
        ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
        ap.add_argument("--lang", default="en")
        ap.add_argument("--batch_size", type=int, default=8)
        ap.add_argument("--images", nargs="*", help="Image paths")
        ap.add_argument("--json", help='Path to JSON file: {"images": [...]}')
        ap.add_argument("--save_json_dir", help="res.save_to_json output dir")
        ap.add_argument("--save_img_dir", help="res.save_to_img output dir")
        args = ap.parse_args()

        # gather images
        images: List[str] = []
        if args.json:
            with open(args.json, "r", encoding="utf-8") as f:
                images.extend(map(str, (json.load(f).get("images") or [])))
        if args.images:
            images.extend(args.images)
        images = [p for p in images if p and os.path.exists(p)]
        if not images:
            print(json.dumps({"results": {}, "errors": {"_general": "No valid images"}}), flush=True)
            return

        # create OCR
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=args.lang,
            device=args.device,
        )

        # inference first (keep GPU busy), then save files
        try:
            results = ocr.predict(images, batch_size=args.batch_size)
        except TypeError:
            # older builds may not expose batch_size on this call
            results = ocr.predict(images)

        if not isinstance(results, (list, tuple)):
            results = [results]
        if len(results) == 1 and isinstance(results[0], (list, tuple)):
            results = list(results[0])

        # summarize
        for img, res in zip(images, results):
            jd = getattr(res, "json", {}) or {}
            n = len(jd.get("rec_texts") or []) if isinstance(jd, dict) else 0
            payload["results"][img] = {"count": n, "device": args.device}

        # save AFTER inference to avoid starving the GPU
        if args.save_json_dir or args.save_img_dir:
            for img, res in zip(images, results):
                try:
                    if args.save_img_dir:
                        res.save_to_img(args.save_img_dir)
                    if args.save_json_dir:
                        res.save_to_json(args.save_json_dir)
                except Exception as e:
                    payload.setdefault("errors", {})[img] = f"SaveError: {e}"

        print(json.dumps(payload, ensure_ascii=False), flush=True)

    except Exception as e:
        payload.setdefault("errors", {})["_fatal"] = f"{type(e).__name__}: {e}"
        print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _service_main()
