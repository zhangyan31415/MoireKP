import os, time, sys, math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ------------------ 纯 Python 计算核（故意避免向量化/内建加速） ------------------
def work(n: int) -> int:
    acc = 0
    for i in range(1, n):
        acc += (i * i) % 97
        acc ^= (acc << 1) & 0xFFFFFFFF
        acc = (acc * 2654435761) & 0xFFFFFFFF
    return acc

# 目标是让单次 work 的时间 ~0.25s，自动粗略校准 n，保证不同机器上输出对比清晰
def calibrate(target=0.25, n0=200_000):
    n = n0
    for _ in range(4):
        t0 = time.perf_counter(); work(n); t = time.perf_counter() - t0
        if t <= 1e-6: t = 1e-6
        n = max(50_000, int(n * (target / t)))
    return n

def bench_threads(n: int, threads: int, tasks: int) -> float:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(work, [n] * tasks))
    return time.perf_counter() - t0

def bench_processes(n: int, procs: int, tasks: int) -> float:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=procs) as ex:
        list(ex.map(work, [n] * tasks))
    return time.perf_counter() - t0

def fmt_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

if __name__ == "__main__":
    pyver = sys.version.split()[0]
    gil_flag = getattr(sys, "_is_gil_enabled", lambda: None)()
    print(f"[Python] {pyver}  |  free-threaded? {'yes' if 'free' in sys.version.lower() else 'no'}")
    print(f"[GIL]    enabled? -> {gil_flag}")
    print(f"[Hint]   用 -X gil=0 可强制关闭（前提是 free-threaded 解释器）。\n")

    # 为了直观，选择几档线程数：1, 2, 4, 8, (CPU 总核)
    cpus = max(1, os.cpu_count() or 1)
    candidates = sorted(set([1, min(2, cpus), min(4, cpus), min(8, cpus), cpus]))
    # 自动校准单次任务规模
    n = calibrate()
    print(f"[Calib]  已校准单次任务规模 n={n}（~0.25s/任务，视机器而定）")
    print(f"[CPU]    detected cores: {cpus}\n")

    # 为避免 BLAS/OpenMP 干扰，这里只做纯 Python；进程并行作对照
    widths = [6, 7, 8, 10, 9]
    print(fmt_row(["Mode", "Workers", "Tasks", "Time(s)", "Speedup"], widths))
    print(fmt_row(["-"*4, "-"*7, "-"*8, "-"*10, "-"*9], widths))

    # 线程基线（1 线程）
    tasks_per_worker = 2  # 每个 worker 跑两个任务，负载随并发增长
    t1 = bench_threads(n, 1, tasks_per_worker * 1)
    print(fmt_row(["thread", 1, tasks_per_worker * 1, f"{t1:.3f}", f"{1.00:.2f}x"], widths))
    for th in candidates[1:]:
        dt = bench_threads(n, th, tasks_per_worker * th)
        print(fmt_row(["thread", th, tasks_per_worker * th, f"{dt:.3f}", f"{t1/dt:.2f}x"], widths))

    print()
    # 进程并行（作参考）
    p1 = bench_processes(n, 1, tasks_per_worker * 1)
    print(fmt_row(["proc", 1, tasks_per_worker * 1, f"{p1:.3f}", f"{1.00:.2f}x"], widths))
    for pr in candidates[1:]:
        dt = bench_processes(n, pr, tasks_per_worker * pr)
        print(fmt_row(["proc", pr, tasks_per_worker * pr, f"{dt:.3f}", f"{p1/dt:.2f}x"], widths))

    print("\n[Note]  如果在“thread”行里，GIL=1 时几乎不加速，而 GIL=0 时随线程变快，就证明无 GIL 生效。")
    print("        进程（proc）一般两种模式都能随并发变快，但有额外的跨进程开销。")
