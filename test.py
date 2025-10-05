import sys, logging
logging.basicConfig(level=logging.INFO)

logging.info("[TORCH] imported? %s", 'torch' in sys.modules)
try:
    import torch
    logging.info("[TORCH] version=%s  cuda.is_available=%s  cuda.is_initialized=%s",
                 torch.__version__, torch.cuda.is_available(),
                 getattr(torch.cuda, "is_initialized", lambda: "n/a")())
except Exception as e:
    logging.info("[TORCH] not importable: %r", e)


import paddle
print("a")
print(paddle.device.get_device())
print("b")
logging.info("[PADDLE] device=%s  is_compiled_with_cuda=%s",
             paddle.device.get_device(), paddle.is_compiled_with_cuda())

