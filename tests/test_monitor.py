"""run_mcmc(monitor=RunMonitor(...)): chunked sampling gives the same draws, and the files the
viewer reads are written and served."""
import json
import os
import threading
import urllib.error
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bananafit  # noqa: F401  (x64)
from bananafit.monitor import RunMonitor, make_server, text_summary
from bananafit.sampling import run_mcmc

D = 4
_rng = np.random.default_rng(0)
_A = _rng.normal(size=(D, D))
COV = _A @ _A.T / D + np.eye(D) * 0.3
PREC = jnp.asarray(np.linalg.inv(COV))
MU = jnp.asarray([0.4, -0.2, 0.1, 0.0])


def logdensity(z):
    d = z - MU
    return -0.5 * d @ PREC @ d


def hop(z):  # an involution: mirror the first coordinate about 0.3
    return z.at[0].set(0.6 - z[0])


KERNELS = {
    "rwm": dict(algorithm="rwm", proposal_cov=jnp.asarray(COV), thinning=2),
    "rwm_hop_blocks": dict(algorithm="rwm", proposal_cov=jnp.asarray(COV), hop_prob=0.1, hop_move_fn=hop,
                           block_moves=[dict(name="b01", prob=0.3, indices=(0, 1), cov=COV[:2, :2])]),
    "nuts": dict(algorithm="nuts", inverse_mass_matrix=jnp.asarray(COV), warmup="step-size"),
    "nuts_hop": dict(algorithm="nuts", inverse_mass_matrix=jnp.asarray(COV), warmup="step-size",
                     hop_prob=0.2, hop_move_fn=hop),
    "mala": dict(algorithm="mala"),
}


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_monitored_run_gives_the_same_draws(tmp_path, kernel):
    kw = dict(n_chains=3, n_warmup=30, n_samples=23, seed=5, **KERNELS[kernel])
    plain = run_mcmc(logdensity, jnp.zeros(D), **kw)
    mon = RunMonitor(tmp_path / kernel, every=10)          # chunks of 10, 10, 3
    watched = run_mcmc(logdensity, jnp.zeros(D), monitor=mon, **kw)
    np.testing.assert_array_equal(np.asarray(watched.positions), np.asarray(plain.positions))
    np.testing.assert_array_equal(np.asarray(watched.log_prob), np.asarray(plain.log_prob))
    assert set(watched.info) == set(plain.info)
    for k in plain.info:
        np.testing.assert_array_equal(np.asarray(watched.info[k]), np.asarray(plain.info[k]))
    chunks = sorted(os.listdir(tmp_path / kernel / "chunks"))
    assert chunks == ["chunk_00000.npz", "chunk_00001.npz", "chunk_00002.npz"]
    on_disk = np.concatenate([np.load(tmp_path / kernel / "chunks" / c)["positions"] for c in chunks], axis=1)
    np.testing.assert_array_equal(on_disk, np.asarray(plain.positions))


def test_monitor_files_summary_and_text(tmp_path):
    d = tmp_path / "fit"
    mon = RunMonitor(d, every=50, names=["a", "b", "c", "d"], watch=["a", "c"],
                     derived={"P(a > 0.4)": lambda z: (z[0] > 0.4).astype(jnp.float64)}, max_trace_points=40)
    run = run_mcmc(logdensity, jnp.zeros(D), algorithm="nuts", n_chains=4, n_warmup=100, n_samples=200,
                   seed=0, inverse_mass_matrix=jnp.asarray(COV), warmup="step-size", monitor=mon)
    meta = json.load(open(d / "meta.json"))
    summary = json.load(open(d / "summary.json"))
    trace = json.load(open(d / "trace.json"))
    assert meta["status"] == "finished" and meta["name"] == "fit" and meta["watch"] == ["a", "c"]
    assert meta["tuned"]["step_size"] is not None
    assert summary["draws_done"] == summary["draws_total"] == 200
    assert len(summary["history"]) == 4 and [r["draws"] for r in summary["history"]] == [50, 100, 150, 200]
    means = {p["name"]: p["mean"] for p in summary["params"]}
    np.testing.assert_allclose([means[n] for n in "abcd"], np.asarray(run.positions).reshape(-1, D).mean(0), rtol=1e-5)
    assert all(p["ess"] > 50 and p["rhat"] < 1.05 for p in summary["params"])
    (dq,) = summary["derived"]
    assert dq["name"] == "P(a > 0.4)" and abs(dq["mean"] - 0.5) < 0.15 and dq["mcse"] > 0
    assert set(trace["series"]) == {"a", "c"}
    assert len(trace["x"]) <= 40 and len(trace["series"]["a"]) == 4 and len(trace["log_density"][0]) == len(trace["x"])
    txt = text_summary(d)
    assert "status finished" in txt and "draws 200/200" in txt and "P(a > 0.4)" in txt


def test_monitor_physical_units_with_a_parameter_set(tmp_path):
    from bananafit.monitor import octant_fraction

    samples, ps, truth = bananafit.make_dune_like_analysis(n_events_per_sample=300, n_spline_systs=1,
                                                           n_norm_systs=1, seed=0)
    model = bananafit.build_forward_model(samples, ps)
    post = bananafit.make_posterior(model, bananafit.asimov_data(model, ps, overrides=truth), ps)
    mon = RunMonitor(tmp_path / "toy", every=15, parameter_set=ps, watch=["theta23"],
                     derived={"P(upper octant)": octant_fraction(ps)})
    run = run_mcmc(post.logdensity, post.z0, algorithm="rwm", proposal_cov=1e-4 * jnp.eye(ps.n_free),
                   n_chains=2, n_warmup=20, n_samples=30, seed=1, monitor=mon)
    summary = json.load(open(tmp_path / "toy" / "summary.json"))
    th = next(p for p in summary["params"] if p["name"] == "theta23")
    i = list(ps.names).index("theta23")
    phys = jax.vmap(ps.constrain)(jnp.asarray(run.positions).reshape(-1, ps.n_free))[:, i]
    np.testing.assert_allclose(th["mean"], float(phys.mean()), rtol=1e-5)
    assert 0 <= summary["derived"][0]["mean"] <= 1


def test_monitor_marks_a_run_that_stops_early(tmp_path):
    class Failing(RunMonitor):
        def update(self, *a):
            super().update(*a)
            if self._chunk == 2:
                raise RuntimeError("node lost")

    mon = Failing(tmp_path / "x", every=5)
    with pytest.raises(RuntimeError, match="node lost"):
        run_mcmc(logdensity, jnp.zeros(D), algorithm="rwm", proposal_cov=jnp.asarray(COV), n_chains=2,
                 n_warmup=10, n_samples=30, seed=0, monitor=mon)
    assert json.load(open(tmp_path / "x" / "meta.json"))["status"] == "stopped"
    assert len(os.listdir(tmp_path / "x" / "chunks")) == 2
    assert "stopped" in text_summary(tmp_path / "x")


def test_monitor_rejects_bad_settings(tmp_path):
    with pytest.raises(ValueError):
        RunMonitor(tmp_path, every=0)
    with pytest.raises(ValueError, match="watch"):
        run_mcmc(logdensity, jnp.zeros(D), algorithm="rwm", proposal_cov=jnp.asarray(COV), n_chains=2,
                 n_warmup=5, n_samples=5, monitor=RunMonitor(tmp_path, every=2, watch=["nope"]))
    with pytest.raises(TypeError):
        run_mcmc(logdensity, jnp.zeros(D), algorithm="rwm", proposal_cov=jnp.asarray(COV), n_chains=2,
                 n_warmup=5, n_samples=5, monitor=object())


def test_server_serves_the_page_and_only_the_summary_files(tmp_path):
    d = tmp_path / "srv"
    run_mcmc(logdensity, jnp.zeros(D), algorithm="rwm", proposal_cov=jnp.asarray(COV), n_chains=2,
             n_warmup=10, n_samples=20, seed=0, monitor=RunMonitor(d, every=10))
    open(d / "secret.txt", "w").write("x")
    httpd = make_server(d, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via an HTTP proxy
    try:
        page = opener.open(base).read().decode()
        assert "<title>" in page and 'name="color-scheme"' in page
        assert json.loads(opener.open(base + "summary.json").read())["draws_done"] == 20
        assert json.loads(opener.open(base + "meta.json").read())["status"] == "finished"
        for bad in ("secret.txt", "chunks/chunk_00000.npz", "../srv/meta.json"):
            with pytest.raises(urllib.error.HTTPError):
                opener.open(base + bad)
    finally:
        httpd.shutdown()
        httpd.server_close()
