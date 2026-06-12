from types import SimpleNamespace

import tapw.cal_ham_01 as cal_ham_01
from tapw.config import ComputeConfig


def _make_cfg(**overrides):
    cfg = SimpleNamespace(
        TAPW=True,
        parallel_impl="joblib",
        parallel_backend="loky",
        num_processes=16,
        tapw_auto_fork=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_resolve_kpoint_parallel_policy_auto_promotes_tapw_defaults_to_mp():
    helper = getattr(cal_ham_01, "resolve_kpoint_parallel_policy", None)
    assert helper is not None

    policy = helper(_make_cfg(), os_name="posix")

    assert policy.parallel_impl == "mp"
    assert policy.parallel_backend == "loky"
    assert policy.auto_promoted is True


def test_resolve_kpoint_parallel_policy_respects_explicit_opt_out():
    helper = getattr(cal_ham_01, "resolve_kpoint_parallel_policy", None)
    assert helper is not None

    policy = helper(_make_cfg(tapw_auto_fork=False), os_name="posix")

    assert policy.parallel_impl == "joblib"
    assert policy.parallel_backend == "loky"
    assert policy.auto_promoted is False


def test_resolve_kpoint_parallel_policy_keeps_non_tapw_joblib_loky():
    helper = getattr(cal_ham_01, "resolve_kpoint_parallel_policy", None)
    assert helper is not None

    policy = helper(_make_cfg(TAPW=False), os_name="posix")

    assert policy.parallel_impl == "joblib"
    assert policy.parallel_backend == "loky"
    assert policy.auto_promoted is False


def test_resolve_kpoint_parallel_policy_requires_multiple_processes_and_posix():
    helper = getattr(cal_ham_01, "resolve_kpoint_parallel_policy", None)
    assert helper is not None

    one_proc = helper(_make_cfg(num_processes=1), os_name="posix")
    non_posix = helper(_make_cfg(), os_name="nt")

    assert one_proc.parallel_impl == "joblib"
    assert one_proc.auto_promoted is False
    assert non_posix.parallel_impl == "joblib"
    assert non_posix.auto_promoted is False


def test_compute_config_disables_tapw_auto_fork_by_default():
    cfg = ComputeConfig()

    assert cfg.tapw_auto_fork is False


def test_compute_config_accepts_zero_blas_threads_as_no_limit():
    cfg = ComputeConfig(blas_threads=0)

    assert cfg.blas_threads == 0
