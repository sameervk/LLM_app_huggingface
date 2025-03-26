## 2025-03-26

- System configuration
  - Intel Core i5 @ 2.30 GHz x 4 processors
  - GPU: NVIDIA GeForce GTX 960M | 2 GB Memory
  - 32 GB RAM

- Embedding dim: 768 gives the following error
  - "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 1.95 GiB of which 7.19 MiB is free. Including non-PyTorch memory, this process has 1.94 GiB memory in use. Of the allocated memory 1.80 GiB is allocated by PyTorch, and 91.74 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)"
  - Works fine for 512 size
  - Time taken to run `train.py` for 1 epoch:
    - CPU: 22.9 s
    - GPU: 7.4 s
  - See `training_parameters.yaml` and `GPT2_arch_config.yaml` for other parameters