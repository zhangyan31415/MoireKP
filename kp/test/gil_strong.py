import os, time, sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def work(n: int) -> int:
    acc = 0
    for i in range(1, n):
        acc += (i * i) % 97
        acc ^= (acc << 1) & 0xFFFFFFFF
        acc = (acc * 2654435761) & 0xFFFFFFFF
    return acc

def calibrate(target=0.30, n0=200_000):
    n = n0
    for _ in range(4):
        t0 = time.perf_counter(); work(n); dt = time.perf_counter() - t0
        if dt <= 1e-6: dt = 1e-6
        n = max(50_000, int(n * (target / dt)))
    return n

def bench_threads(n, threads, total_tasks):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        ex.map(work, [n]*total_tasks)
    return time.perf_counter() - t0

def bench_processes(n, procs, total_tasks):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=procs) as ex:
        ex.map(work, [n]*total_tasks)
    return time.perf_counter() - t0

def row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

if __name__ == "__main__":
    ver = sys.version.split()[0]
    gil = getattr(sys, "_is_gil_enabled", lambda: None)()
    print(f"[Python] {ver} | free-threaded? {'yes' if 'free' in sys.version.lower() else 'no'}")
    print(f"[GIL]    enabled? -> {gil}\n")

    # 固定总任务数（强缩放），让 1 线程下总时长 ~3s，易观察
    n = calibrate(target=0.30)
    total_tasks = 10  # 总任务固定
    print(f"[Calib] 单任务规模 n={n}（~0.30s/任务）; 总任务={total_tasks}")
    cpus = max(1, os.cpu_count() or 1)
    steps = [1, 2, 4, 8, 16]
    steps = [s for s in steps if s <= cpus]
    print(f"[CPU]   detected cores: {cpus}\n")

    widths = [6, 7, 8, 10, 8, 9]
    print(row(["Mode","Workers","Tasks","Time(s)","Speedup","Eff(%)"], widths))
    print(row(["----","-------","-----","--------","-------","------"], widths))

    t1 = bench_threads(n, 1, total_tasks)
    print(row(["thread",1,total_tasks,f"{t1:.3f}", "1.00x", f"{100:.1f}"], widths))
    for th in steps[1:]:
        dt = bench_threads(n, th, total_tasks)
        sp = t1/dt
        eff = sp/th*100
        print(row(["thread",th,total_tasks,f"{dt:.3f}", f"{sp:.2f}x", f"{eff:.1f}"], widths))

    print()
    p1 = bench_processes(n, 1, total_tasks)
    print(row(["proc",1,total_tasks,f"{p1:.3f}", "1.00x", f"{100:.1f}"], widths))
    for pr in steps[1:]:
        dt = bench_processes(n, pr, total_tasks)
        sp = p1/dt
        eff = sp/pr*100
        print(row(["proc",pr,total_tasks,f"{dt:.3f}", f"{sp:.2f}x", f"{eff:.1f}"], widths))

    print("\n[Interpret]")
    print("  • GIL=1：thread 区通常 Speedup≈1（甚至低于1）;")
    print("  • GIL=0：thread 随并发 Time 应明显下降，Speedup 上升，Eff(%) 接近 50%~80%（取决于核数/频率/竞争）。")
