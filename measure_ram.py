import subprocess, psutil, sys, time

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "runs/baseline/checkpoint.pt"

cmd = [
    sys.executable, "evaluate.py",
    "--checkpoint", checkpoint,
    "--device", "cpu",
    "--precision", "fp32",
    "--split", "test",
]

proc = subprocess.Popen(cmd)
p = psutil.Process(proc.pid)
peak = 0
while proc.poll() is None:
    try:
        mem = p.memory_info().rss
        peak = max(peak, mem)
    except psutil.NoSuchProcess:
        break
    time.sleep(0.05)

print(f"Peak RSS: {peak / 1024**3:.3f} GiB")