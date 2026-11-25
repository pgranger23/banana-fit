"""Follow an MCMC run while it goes.

Pass a :class:`RunMonitor` to :func:`bananafit.sampling.run_mcmc`::

    from bananafit.monitor import RunMonitor, octant_fraction

    mon = RunMonitor("runs/fit1", every=500, parameter_set=ps,
                     watch=["theta23", "dm31", "deltacp"],
                     derived={"P(upper octant)": octant_fraction(ps)})
    run = run_mcmc(post.logdensity, z0, ..., monitor=mon)

The sampling phase then runs in chunks of ``every`` stored draws. After each
chunk the monitor writes into its directory:

- ``chunks/chunk_NNNNN.npz``: that chunk's draws (positions, log density, sampler info), so a
  run that is killed keeps everything up to its last chunk;
- ``summary.json``: progress, time left, running means and widths, R-hat and ESS of every
  parameter, the derived quantities with their Monte Carlo errors, and one row per chunk
  (acceptance, divergences, tree depth, convergence so far);
- ``trace.json``: thinned traces of the watched parameters and of the log density;
- ``meta.json``: the run's settings and its status (running, finished, stopped).

Watch it with ``python -m bananafit.monitor runs/fit1``, which serves a page that refreshes
itself (``--text`` prints the summary instead). For a run on a cluster node, open an ssh tunnel
to the port, or copy the directory back and point the viewer at the copy.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
import warnings

import numpy as np

__all__ = ["RunMonitor", "octant_fraction", "text_summary", "snapshot", "make_server", "serve", "main"]

_FILES = ("meta.json", "summary.json", "trace.json")


def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path, obj):
    """Write atomically (temporary file + rename), so a reader never sees half a file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def _clean(x, digits=6):
    """JSON-safe floats: NaN/inf -> None, rounded to ``digits`` significant digits."""
    if isinstance(x, dict):
        return {k: _clean(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, digits) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist(), digits)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{digits}g}")
    return x


def octant_fraction(parameter_set, name: str = "theta23", threshold: float | None = None):
    """Derived quantity for :class:`RunMonitor`: the indicator that the mixing angle is in the
    upper octant, so its running mean is P(upper octant).

    ``threshold`` defaults to pi/4, for an angle in radians (the package convention). Pass
    0.5 for a parameter that holds sin^2(theta23).
    """
    import jax.numpy as jnp

    idx = list(parameter_set.names).index(name)
    thr = float(np.pi / 4 if threshold is None else threshold)

    def indicator(z):
        return (parameter_set.constrain(z)[idx] > thr).astype(jnp.float64)

    return indicator


class RunMonitor:
    """Writes a run's draws and summary to ``directory`` chunk by chunk.

    Parameters
    ----------
    directory:
        where the files go (created if needed; files from an earlier run there are replaced).
    every:
        stored draws per chunk (per chain). Each chunk costs one device-to-host copy, so keep a
        chunk at least tens of seconds long.
    names:
        names of the free parameters (default: ``parameter_set.free_names``, else ``z0``, ``z1``...).
    parameter_set:
        optional; with it, means, widths and traces are in physical units (``constrain``), and
        circular parameters get the circular R-hat/ESS of :mod:`bananafit.diagnostics`.
    watch:
        free-parameter names whose traces are kept for the viewer (default: the first six).
    derived:
        ``{label: f}`` with ``f(z) -> scalar`` (jax-traceable), summarised as a running mean with
        its Monte Carlo error, e.g. :func:`octant_fraction`.
    max_trace_points:
        points per chain kept in ``trace.json`` (the traces are thinned to fit).
    save_draws:
        write each chunk's draws to ``chunks/`` (default ``True``).
    """

    def __init__(self, directory, every: int = 1000, *, names=None, parameter_set=None,
                 watch=None, derived=None, max_trace_points: int = 1000, save_draws: bool = True):
        self.directory = os.fspath(directory)
        self.every = int(every)
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.parameter_set = parameter_set
        self._names_arg = list(names) if names is not None else None
        self._watch_arg = list(watch) if watch is not None else None
        self.derived = dict(derived or {})
        self.max_trace_points = int(max_trace_points)
        self.save_draws = bool(save_draws)

    # -------------------------------------------------------------- called by run_mcmc
    def start(self, *, algorithm, n_chains, n_samples, thinning, n_free, tuned=None):
        import jax

        ps = self.parameter_set
        if self._names_arg is not None:
            names = self._names_arg
        elif ps is not None:
            names = list(ps.free_names)
        else:
            names = [f"z{i}" for i in range(n_free)]
        if len(names) != n_free:
            raise ValueError(f"{len(names)} names for {n_free} free parameters")
        watch = self._watch_arg if self._watch_arg is not None else names[:6]
        unknown = [w for w in watch if w not in names]
        if unknown:
            raise ValueError(f"watch names not among the free parameters: {unknown}")
        self.names, self.watch = names, watch
        self._watch_idx = [names.index(w) for w in watch]
        from bananafit.diagnostics import _resolve_circular_names

        circular = set(_resolve_circular_names(ps, None))
        self._circ = [j for j, n in enumerate(names) if n in circular]
        if ps is not None:
            all_names = list(ps.names)
            self._phys_idx = [all_names.index(n) for n in names]
            constrain = jax.jit(jax.vmap(ps.constrain))
            to_phys = lambda z: np.array(constrain(z))[:, self._phys_idx]  # noqa: E731
        else:
            to_phys = lambda z: np.array(z)  # noqa: E731

        def wrapped(z):  # circular parameters are sampled unwrapped: show them in (-pi, pi]
            x = to_phys(z)
            if self._circ:
                x[:, self._circ] = np.pi - np.mod(np.pi - x[:, self._circ], 2 * np.pi)
            return x

        self._to_phys = wrapped
        self._derived_fns = {k: jax.jit(jax.vmap(f)) for k, f in self.derived.items()}
        self.algorithm, self.n_chains, self.n_samples, self.thinning = algorithm, n_chains, n_samples, thinning
        self._pos, self._lp, self._phys, self._der = [], [], [], {k: [] for k in self.derived}
        self._history, self._done, self._chunk = [], 0, 0
        self._divergent_total = 0
        os.makedirs(self.directory, exist_ok=True)
        if self.save_draws:
            os.makedirs(os.path.join(self.directory, "chunks"), exist_ok=True)
            for f in os.listdir(os.path.join(self.directory, "chunks")):
                if f.startswith("chunk_") and f.endswith(".npz"):
                    os.remove(os.path.join(self.directory, "chunks", f))
        for f in _FILES:
            p = os.path.join(self.directory, f)
            if os.path.exists(p):
                os.remove(p)
        tuned_meta = {}
        for k, v in (tuned or {}).items():
            a = np.asarray(v)
            if k == "step_size" or a.size <= 64:
                tuned_meta[k] = a
        try:
            from bananafit import __version__ as version
        except Exception:  # pragma: no cover
            version = None
        self._t0 = time.time()
        self.meta = dict(name=os.path.basename(os.path.abspath(self.directory)),
                         bananafit=version, algorithm=algorithm, n_chains=int(n_chains),
                         n_samples=int(n_samples), thinning=int(thinning), every=self.every,
                         n_free=int(n_free), names=names, watch=watch,
                         derived=list(self.derived), physical=ps is not None,
                         tuned=_clean(tuned_meta), started=_now(), status="running")
        _write_json(os.path.join(self.directory, "meta.json"), self.meta)

    def update(self, positions, log_prob, infos):
        positions = np.asarray(positions)
        log_prob = np.asarray(log_prob)
        infos = {k: np.asarray(v) for k, v in infos.items()}
        nc, m, nf = positions.shape
        flat = positions.reshape(-1, nf)
        self._pos.append(positions)
        self._lp.append(log_prob)
        self._phys.append(self._to_phys(flat).reshape(nc, m, nf))
        for k, f in self._derived_fns.items():
            self._der[k].append(np.asarray(f(flat), dtype=float).reshape(nc, m))
        if self.save_draws:
            np.savez(os.path.join(self.directory, "chunks", f"chunk_{self._chunk:05d}.npz"),
                     positions=positions, log_prob=log_prob,
                     **{f"info_{k}": v for k, v in infos.items()})
        self._chunk += 1
        self._done += m
        self._write_summary(infos)

    def finish(self, complete: bool = True):
        if not hasattr(self, "meta"):
            return
        self.meta["status"] = "finished" if complete else "stopped"
        self.meta["ended"] = _now()
        _write_json(os.path.join(self.directory, "meta.json"), self.meta)

    # -------------------------------------------------------------- summaries
    def _write_summary(self, infos):
        from bananafit import diagnostics as D

        pos = np.concatenate(self._pos, axis=1)
        phys = np.concatenate(self._phys, axis=1)
        lp = np.concatenate(self._lp, axis=1)
        n = pos.shape[1]

        class _R:  # what the diagnostics need from a run
            positions = pos

        rh = es = {}
        if n >= 4:
            try:  # diagnostics are a convenience: never stop the run, never warn about short chains
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rh = D.rhat(_R, names=self.names, parameter_set=self.parameter_set)
                    es = D.ess(_R, names=self.names, parameter_set=self.parameter_set)
            except Exception:
                rh = es = {}
        params = []
        for j, name in enumerate(self.names):
            x = phys[..., j]
            if j in self._circ:  # circular mean and spread sqrt(-2 ln R)
                c, sn = np.cos(x).mean(), np.sin(x).mean()
                r = min(math.hypot(c, sn), 1.0)
                mean, sd = math.atan2(sn, c), math.sqrt(-2 * math.log(r)) if r > 0 else math.inf
            else:
                mean, sd = float(x.mean()), float(x.std())
            params.append(dict(name=name, mean=mean, sd=sd, circular=j in self._circ,
                               rhat=rh.get(name), ess=es.get(name)))
        derived = []
        for k, parts in self._der.items():
            v = np.concatenate(parts, axis=1)
            e = _ess_1d(v)
            mean = float(v.mean())
            mcse = float(v.std() / math.sqrt(e)) if e and e > 0 else None
            derived.append(dict(name=k, mean=mean, mcse=mcse, ess=e,
                                per_chain=v.mean(axis=1).tolist()))
        rhats = [p["rhat"] for p in params if p["rhat"] is not None and math.isfinite(p["rhat"])]
        esss = [p["ess"] for p in params if p["ess"] is not None and math.isfinite(p["ess"])]
        elapsed = time.time() - self._t0
        rate = self._done / elapsed if elapsed > 0 else None
        eta = (self.n_samples - self._done) / rate if rate else None

        row = dict(chunk=self._chunk, draws=self._done, steps=self._done * self.thinning,
                   elapsed_s=elapsed, max_rhat=max(rhats) if rhats else None,
                   min_ess=min(esss) if esss else None)
        if "acceptance_rate" in infos:
            row["acceptance"] = float(infos["acceptance_rate"].mean())
        if "is_divergent" in infos:
            nd = int(np.sum(infos["is_divergent"]))
            row["divergent"] = nd
            self._divergent_total += nd
        if "num_integration_steps" in infos:
            row["leapfrog_mean"] = float(infos["num_integration_steps"].mean())
            row["leapfrog_slowest"] = float(np.asarray(infos["num_integration_steps"]).max(axis=0).mean())
        if "is_hop" in infos and "hop_accepted" in infos:
            nh = float(infos["is_hop"].mean())
            row["hop_acceptance"] = float(infos["hop_accepted"].mean()) / nh if nh > 0 else None
        row["derived"] = {d["name"]: [d["mean"], d["mcse"]] for d in derived}
        self._history.append(row)

        summary = dict(updated=_now(), updated_unix=time.time(), status="running",
                       draws_done=self._done, draws_total=self.n_samples,
                       steps_done=self._done * self.thinning, elapsed_s=elapsed,
                       rate_draws_per_s=rate, eta_s=eta, divergent_total=self._divergent_total,
                       params=params, derived=derived, history=self._history)
        _write_json(os.path.join(self.directory, "summary.json"), _clean(summary))

        stride = max(1, math.ceil(n / self.max_trace_points))
        idx = np.arange(0, n, stride)
        series = {}
        for w, j in zip(self.watch, self._watch_idx):
            x = phys[:, idx, j]
            if j in self._circ:  # centre the 2*pi window on the circular mean, away from the seam
                c = params[j]["mean"]
                x = c + np.pi - np.mod(np.pi - (x - c), 2 * np.pi)
            series[w] = x
        trace = dict(x=((idx + 1) * self.thinning).tolist(), series=series, log_density=lp[:, idx],
                     circular=[w for w, j in zip(self.watch, self._watch_idx) if j in self._circ])
        _write_json(os.path.join(self.directory, "trace.json"), _clean(trace))


def _ess_1d(v):
    """ESS of a (n_chains, n_draws) series; None when it cannot be estimated (constant series)."""
    if v.shape[1] < 4 or np.all(v == v.flat[0]):
        return None
    try:
        from blackjax import diagnostics as bd
        import jax.numpy as jnp

        e = float(np.ravel(bd.effective_sample_size(jnp.asarray(v)[..., None], chain_axis=0,
                                                    sample_axis=1))[0])
        return e if math.isfinite(e) else None
    except Exception:  # pragma: no cover
        return None


# ------------------------------------------------------------------ reading a run directory
def _load(directory):
    out = {}
    for f in _FILES:
        p = os.path.join(directory, f)
        if os.path.exists(p):
            with open(p) as fh:
                out[f.split(".")[0]] = json.load(fh)
    return out


def _fmt_s(s):
    if s is None:
        return "-"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def _fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}g}"


def text_summary(directory) -> str:
    """The run's state as plain text (what ``python -m bananafit.monitor DIR --text`` prints)."""
    d = _load(directory)
    if "meta" not in d:
        return f"{directory}: no run yet (no meta.json)"
    meta, s = d["meta"], d.get("summary")
    lines = [f"{directory}: {meta['algorithm']}, {meta['n_chains']} chains, status {meta['status']}"]
    if not s:
        lines.append("  warmup done, waiting for the first chunk")
        return "\n".join(lines)
    pct = 100 * s["draws_done"] / max(s["draws_total"], 1)
    lines.append(f"  draws {s['draws_done']}/{s['draws_total']} per chain ({pct:.0f} %), "
                 f"elapsed {_fmt_s(s['elapsed_s'])}, left {_fmt_s(s['eta_s'])}, "
                 f"updated {s['updated']}")
    last = s["history"][-1] if s["history"] else {}
    bits = []
    for k, label in (("acceptance", "acceptance"), ("leapfrog_slowest", "leapfrog (slowest)"),
                     ("hop_acceptance", "hop acceptance")):
        if last.get(k) is not None:
            bits.append(f"{label} {last[k]:.3g}")
    if "divergent" in last:
        bits.append(f"divergent {s['divergent_total']} in total")
    if bits:
        lines.append("  last chunk: " + ", ".join(bits))
    for dq in s["derived"]:
        lines.append(f"  {dq['name']}: {_fmt(dq['mean'])} ± {_fmt(dq['mcse'], 2)} (ESS {_fmt(dq['ess'], 3)})")
    params = s["params"]
    watch = set(meta.get("watch", []))
    rows = [p for p in params if p["name"] in watch]
    worst = sorted((p for p in params if p["ess"] is not None), key=lambda p: p["ess"])[:3]
    rows += [p for p in worst if p not in rows]
    w = max([len(p["name"]) for p in rows] + [9])
    lines.append(f"  {'parameter':<{w}}  {'mean':>11}  {'sd':>10}  {'R-hat':>6}  {'ESS':>7}")
    for p in rows:
        flag = "  (lowest ESS)" if p in worst and p["name"] not in watch else ""
        lines.append(f"  {p['name']:<{w}}  {_fmt(p['mean'], 5):>11}  {_fmt(p['sd'], 3):>10}  "
                     f"{_fmt(p['rhat'], 4):>6}  {_fmt(p['ess'], 4):>7}{flag}")
    return "\n".join(lines)


# ------------------------------------------------------------------ live page
def _page():
    from bananafit._monitor_page import PAGE

    return PAGE


def snapshot(directory, out, theme: str | None = None) -> str:
    """Write a static copy of the live page for ``directory`` to ``out``: one HTML file with the
    run's current summary embedded, which opens offline (no server) and can be attached or sent.
    ``theme`` = ``"light"`` or ``"dark"`` forces a colour scheme (default: the viewer's)."""
    data = json.dumps(_load(directory), separators=(",", ":")).replace("</", "<\\/")
    page = _page()
    if theme in ("light", "dark"):
        page = page.replace('<html lang="en">', f'<html lang="en" data-theme="{theme}">', 1)
    page = page.replace("<script>", f"<script>window.SNAPSHOT = {data};</script>\n<script>", 1)
    with open(out, "w") as f:
        f.write(page)
    return os.fspath(out)


def make_server(directory, host: str = "127.0.0.1", port: int = 8765):
    """An HTTP server for the live page of ``directory`` (not started). Only the page and the
    three summary files are served; everything else is a 404."""
    import http.server
    import socketserver

    directory = os.path.abspath(os.fspath(directory))
    page = _page().encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0].lstrip("/")
            if path in ("", "index.html"):
                body, ctype = page, "text/html; charset=utf-8"
            elif path in _FILES:
                f = os.path.join(directory, path)
                body = open(f, "rb").read() if os.path.exists(f) else b"null"
                ctype = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    return Server((host, port), Handler)


def serve(directory, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    """Serve the live page for ``directory`` at ``http://host:port/`` until interrupted."""
    httpd = make_server(directory, host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"bananafit.monitor: {os.path.abspath(os.fspath(directory))} at {url} (Ctrl-C to stop)", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bananafit.monitor",
                                 description="Follow a run written by run_mcmc(monitor=RunMonitor(...)).")
    ap.add_argument("directory")
    ap.add_argument("--text", action="store_true", help="print the summary instead of serving the page")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="with --text: reprint every SECONDS until the run ends")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    ap.add_argument("--html", metavar="OUT", help="write a static snapshot of the page to OUT and exit")
    ap.add_argument("--theme", choices=("light", "dark"), help="with --html: force a colour scheme")
    a = ap.parse_args(argv)
    if a.html:
        print(snapshot(a.directory, a.html, a.theme))
        return 0
    if a.text:
        while True:
            print(text_summary(a.directory), flush=True)
            meta = _load(a.directory).get("meta", {})
            if a.watch <= 0 or meta.get("status") in ("finished", "stopped"):
                return 0
            time.sleep(a.watch)
            print()
    serve(a.directory, a.host, a.port, a.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
