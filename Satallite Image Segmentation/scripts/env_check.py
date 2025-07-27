# save as verify_pytorch_gpu.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Run a small test computation on GPU
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"GPU computation test successful: {z.device}")
    
    # Test computation speed (CPU vs GPU)
    import time
    
    # Test matrix multiplication on CPU
    size = 5000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    start = time.time()
    c = torch.matmul(a, b)
    cpu_time = time.time() - start
    
    # Test same operation on GPU
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    
    # Warm-up run
    torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for GPU operation to complete
    gpu_time = time.time() - start
    
    print(f"\nMatrix multiplication speed test ({size}x{size}):")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.2f}x faster")
else:
    print("CUDA is not available. PyTorch is using CPU only.")
    print("Please check your installation.")