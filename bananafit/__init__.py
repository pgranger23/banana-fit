"""BANANA (Bayesian Autodiff Neutrino Analysis & Nuisance Assessment), package
``bananafit``: a differentiable MCMC fitter for neutrino oscillation analyses.

Event samples + spline systematics -> binned likelihood -> blackjax samplers,
with exact gradients through the whole pipeline (oscillations via MANGO, imported as mango).

Float64 is mandatory (oscillation phases): enforced here, same as mango.
See DESIGN.md for the architecture; each submodule docstring explains its
algorithm choices.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.3.2"

from .events import EventSample, make_sample, make_sample_prebinned
from .splines import SplineTable, build_spline_table
from .norms import NormSet, from_selections
from .oscillation import OscConfig, ConstantMatterOsc, VacuumOsc
from .parameters import (
    ParamSpec,
    ParameterSet,
    Gaussian,
    VonMises,
    Flat,
    MVGaussianBlock,
    osc_param_specs,
)
from .likelihood import (
    poisson_loglike,
    barlow_beeston_lite_loglike,
    gaussian_cov_loglike,
    build_nd_covariance,
)
from .model import SampleModel, ForwardModel, build_forward_model
from .shifts import (
    EnergyComponents,
    EnergyScaleModel,
    EnergyShift,
    FD_DETECTOR_PARAM_NAMES,
    soft_bin_weights,
    hard_bin_index,
    cvn_efficiency_weight,
)
from .posterior import BinnedPosterior, make_posterior, asimov_data
from .sampling import (MCMCRun, run_mcmc, map_fit, laplace_covariance,
                       conditional_modes, octant_mirror_hop)
from .diagnostics import rhat, ess, summary, to_arviz, octant_indicator, lockstep_efficiency
from .io import save_analysis, load_analysis, save_sample, load_sample
from .toy import make_dune_like_analysis, poisson_fluctuate
from .root_import import read_tspline3_knots, spline_table_from_root
from .globes_bridge import samples_from_globes

__all__ = [
    "EventSample", "make_sample", "make_sample_prebinned",
    "SplineTable", "build_spline_table",
    "NormSet", "from_selections",
    "OscConfig", "ConstantMatterOsc", "VacuumOsc",
    "ParamSpec", "ParameterSet", "Gaussian", "VonMises", "Flat", "MVGaussianBlock",
    "osc_param_specs",
    "poisson_loglike", "barlow_beeston_lite_loglike",
    "gaussian_cov_loglike", "build_nd_covariance",
    "SampleModel", "ForwardModel", "build_forward_model",
    "EnergyComponents", "EnergyScaleModel", "EnergyShift", "FD_DETECTOR_PARAM_NAMES",
    "soft_bin_weights", "hard_bin_index", "cvn_efficiency_weight",
    "BinnedPosterior", "make_posterior", "asimov_data",
    "MCMCRun", "run_mcmc", "map_fit", "laplace_covariance",
    "conditional_modes", "octant_mirror_hop",
    "rhat", "ess", "summary", "to_arviz", "octant_indicator", "lockstep_efficiency",
    "save_analysis", "load_analysis", "save_sample", "load_sample",
    "make_dune_like_analysis", "poisson_fluctuate",
    "read_tspline3_knots", "spline_table_from_root",
    "samples_from_globes",
]
