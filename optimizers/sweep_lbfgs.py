# sweep_lbfgs.py
import os, sys, json, time, argparse, subprocess, itertools, csv
from contextlib import redirect_stdout
import io

import tensorflow as tf

# ---- Global TF runtime config (do once, before touching TF objects) ----
# GPU: let TF allocate memory on-demand
try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass  # safe on CPU-only

# Allow TF32 on Ampere+ (helps FP32 matmuls/convs)
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

print("TF", tf.__version__)
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

MODULE = "new_lbfgs_all_tol"  # <-- set to your module name (no .py)

OLD_COMBOS = list(itertools.product(
    ["cpu", "gpu"],  # device
    [False, True],  # eager
    [False, True],  # xla
    ["fp64", "hybrid", "fp32"],  # mode
    ["mixed", "float32"]  # model's precision
))

COMBOS = list(itertools.product(
    ["gpu", "cpu"],  # device
    [False],  # eager
    [True, False],  # xla
    ["fp64", "hybrid", "fp32"],  # mode
    ["mixed", "float32"]  # model's precision
))


def run_child(args):
    """
    Executes a single run in this process. We keep this small; parent creates
    fresh processes per combo to isolate device/XLA state.
    """
    import importlib
    import numpy as np
    import tensorflow as tf
    mod = importlib.import_module(MODULE)

    # Determinism & threads
    mod.set_global_determinism(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    # Device setup
    if args.device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

    # Eager/Graph toggle
    tf.config.run_functions_eagerly(bool(args.eager))

    # Optional: device placement logs for one debug run
    if getattr(args, "log_placement", False):
        tf.debugging.set_log_device_placement(True)

    # Make sure variables are created on the intended device
    dev_str = "/GPU:0" if args.device == "gpu" else "/CPU:0"
    with tf.device(dev_str):
        # Data & model (model stays FP32 as in your demos)
        X, Y = mod.make_data(args.N, seed=args.seed, dtype=tf.float32)
        mod.set_model_precision(mod.Precision.MIXED if args.model_precision == "mixed" else mod.Precision.FP32)
        model = mod.build_mlp(seed=args.seed,
                              output_dtype='float64' if args.model_precision == 'float64' else 'float32')
        vars_list = model.trainable_variables

        def loss_fn():
            pred = tf.cast(model(X, training=False), Y.dtype)
            e = tf.cast(Y, pred.dtype) - pred
            return tf.reduce_mean(tf.square(e))

        # Adapter store dtype by mode
        store_dtype = tf.float64 if args.mode == "fp64" else tf.float32

        adapter = mod.LossAdapter(loss_fn=loss_fn, params=vars_list, mode="assign",
                                  opt_dtype=store_dtype, use_xla=bool(args.xla))

        # Config — Option A for fp32(strict) already inside your module via num_tol
        cfg = mod.LBFGSConfig(
            max_iters=args.warmup + args.measure,
            mem=args.mem,
            print_every=10 ** 9,  # suppress inner prints; we’ll print summary here
            chunk_size=args.chunk,
            use_adam_h0=True, beta2=0.999, diag_eps=0.0,
            alpha_init=1.0, ls_max_steps=16, c1=1e-4, cub_clip=(0.1, 2.5),
            powell_c=0.2,
            lbfgs_mode=args.mode,
            fp32_strict=True,
            debug_print=False,
            run_device=args.device
        )
        opt = mod.LBFGS(adapter, cfg, lbfgs_mode=args.mode, use_xla=bool(args.xla))

        # Warm-up (not timed, suppress prints)
        if args.warmup > 0:
            with redirect_stdout(io.StringIO()):
                opt.minimize_chunked(total_iters=args.warmup, chunk_size=args.chunk, print_every=10 ** 9)

        # Timed segment
        t0 = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            opt.minimize_chunked(total_iters=args.measure, chunk_size=args.chunk, print_every=10 ** 9)
        elapsed = time.perf_counter() - t0

        result = {
            "device": args.device,
            "eager": bool(args.eager),
            "xla": bool(args.xla),
            "mode": args.mode,
            "model": args.model_precision,
            "iters": args.measure,
            "seed": args.seed,
            "loss": float(opt.f.numpy()),
            "time_sec": float(elapsed),
            "mem_len": int(opt.mem.len.numpy()),
        }
        print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help="Run a single child job")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--eager", type=int, choices=[0, 1], default=0)
    parser.add_argument("--xla", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mode", choices=["fp64", "hybrid", "fp32"], default="hybrid")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--N", type=int, default=32768 * 2)
    parser.add_argument("--iters", type=int, default=1000, help="(deprecated) use --measure")
    parser.add_argument("--measure", type=int, default=1000, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=50, help="non-timed warmup iters")
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--mem", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model_precision", choices=["mixed", "float32"], default="float32")
    args = parser.parse_args()

    if args.child:
        # CPU visibility should be handled by parent via env before TF loads.
        run_child(args)
        return

    # Parent sweeps all combos via subprocesses
    results = []
    print("Running 24-run sweep...\n")
    for device, eager, xla, mode, model_precision in COMBOS:
        tf.debugging.set_log_device_placement(True)
        # Skip impossible combo: XLA in eager is supported, but slower; we still include it per your request.
        env = os.environ.copy()
        if device == "cpu":
            # Hide GPUs *before* TF loads in the child
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)  # allow default GPU visibility

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--child",
            "--device", device,
            "--eager", "1" if eager else "0",
            "--xla", "1" if xla else "0",
            "--mode", mode,
            "--seed", str(args.seed),
            "--N", str(args.N),
            "--measure", str(args.measure),
            "--warmup", str(args.warmup),
            "--chunk", str(args.chunk),
            "--mem", str(args.mem),
            "--threads", str(args.threads),
            "--model_precision", model_precision
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"[ERROR] device={device} eager={int(eager)} xla={int(xla)} mode={mode} model={model_precision}:")
            print(proc.stderr)
            continue
        # Child prints exactly one JSON line
        line = proc.stdout.strip().splitlines()[-1]
        try:
            rec = json.loads(line)
            results.append(rec)
            print(f"{device:>3} | eager={int(eager)} | xla={int(xla)} | {mode:>5} | {model_precision:>5} "
                  f"=> loss={rec['loss']:.6g}  time={rec['time_sec']:.3f}s  mem={rec['mem_len']}")
        except Exception:
            print(f"[WARN] Could not parse child output:\n{proc.stdout}\n{proc.stderr}")

    # Save CSV + pretty table
    if results:
        csv_path = "lbfgs_sweep_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["device", "eager", "xla", "mode", "iters", "seed", "loss", "time_sec",
                                              "mem_len"])
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nSaved: {csv_path}")

        # Pretty summary grouped by device/mode
        def key(r):
            return r["device"], r["mode"], int(r["eager"]), int(r["xla"])

        results.sort(key=lambda r: (r["device"], r["mode"], r["time_sec"]))
        print("\nSummary (sorted by device/mode/time):")
        print("device mode   eager xla   time[s]   loss        mem")
        for r in results:
            print(f"{r['device']:>3}    {r['mode']:>5}    {int(r['eager']):>1}     {int(r['xla']):>1}   "
                  f"{r['time_sec']:>7.3f}  {r['loss']:.3e}  {r['mem_len']:>3}")


if __name__ == "__main__":
    main()
