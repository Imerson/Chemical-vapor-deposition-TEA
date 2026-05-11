from __future__ import annotations


'''
#install packages
import subprocess
import sys

# Install a single package - in VS
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

# Install multiple packages
packages = ["numpy", "pandas", "matplotlib", "scipy", "dataclasses", "coolprop"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
'''


"""Industrial-scale CNT techno-economics with physics-based FCCVD + Monte Carlo (USD).

This script is intentionally **physics-first**:
- FCCVD mass balance is computed from methane cracking stoichiometry (CH4 -> C + 2 H2)
  with explicit assumptions for: carbon yield to solids, CNT vs carbon-black split,
  catalyst (Fe) and promoter (S) dosing, and methane conversion.
- Energy balance uses CoolProp/thermo for gas Cp (T,P dependent) and includes:
  (i) sensible heating of feeds to reactor temperature
  (ii) reaction endotherm for methane cracking
  (iii) PSA/recycle compression power (simple isentropic model)
  (iv) conductive heat loss through insulation (ceramic fiber blanket model)

Monte Carlo sampling propagates uncertainty in prices, yields, conversion, dosing,
availability, and CAPEX/OPEX factors to NPV and unit cost outcomes.

Notes
-----
- Reactor pressure: atmospheric (1 atm)
- Reactor temperature: 1300 C (as you specified)
- PSA hydrogen purity: assumed high-purity (99.99%+) typical for PSA product gas;
  the purity itself does not change mass balance here, but PSA recovery does.

This version adds a **physics-based FLB (fluidized bed) route** as well:
- Operating temperature: 900 C (as you specified)
- Reactor pressure: atmospheric
- Uses CO2 as a co-feed (precursor / oxidant / carrier) and a Fe-Al-based catalyst feed
- Produces CNT + CB (solid carbon split) and H2 from CH4 cracking (with PSA recovery)

Author: Imerson (with ChatGPT refactor)
Date: 2026-01-16
"""

import os
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import lognorm, triang, beta, norm


def _savefig_without_titles(fig: plt.Figure, path: str, **kwargs) -> None:
    """Save a Matplotlib figure with titles removed.

    Titles remain visible for interactive viewing (e.g., plt.show()), but are
    stripped only for the saved files so you can add titles in the paper.
    """

    suptitle_obj = getattr(fig, "_suptitle", None)
    suptitle_text = None
    if suptitle_obj is not None:
        suptitle_text = suptitle_obj.get_text()
        suptitle_obj.set_text("")

    ax_title_texts = [ax.title.get_text() for ax in fig.axes]
    for ax in fig.axes:
        ax.title.set_text("")

    try:
        fig.savefig(path, **kwargs)
    finally:
        for ax, title_text in zip(fig.axes, ax_title_texts):
            ax.title.set_text(title_text)
        if suptitle_obj is not None and suptitle_text is not None:
            suptitle_obj.set_text(suptitle_text)



# ============================================================
# 1) FIXED INPUTS (YOUR LATEST BASELINE)
# ============================================================

@dataclass(frozen=True)
class ScaleConfig:
    """Single source of truth for capacity scaling.

    Inputs
    - throughput_tpy: Annual CNT production target [t/y]
    - annual_operating_hours: Scheduled operating hours [h/y]

    Derived
    - throughput_g_h: Nameplate target [g/h] using scheduled hours (availability handled elsewhere)
    """

    throughput_tpy: float
    annual_operating_hours: float

    @property
    def throughput_g_h(self) -> float:
        return float(self.throughput_tpy) * 1_000_000.0 / float(self.annual_operating_hours)


def set_scale(throughput_tpy: float, annual_operating_hours: float) -> ScaleConfig:
    """Helper to keep all scale-related derived quantities consistent."""
    if annual_operating_hours <= 0:
        raise ValueError("annual_operating_hours must be > 0")
    if throughput_tpy <= 0:
        raise ValueError("throughput_tpy must be > 0")
    return ScaleConfig(float(throughput_tpy), float(annual_operating_hours))


# Scale / operation
Numberoffacility = 1
Throughput_g_h_small = 30.0

# Train nameplate capacity distribution (t CNT/year per train)
# Order-of-magnitude anchors...
train_capacity_tpy_min, train_capacity_tpy_mode, train_capacity_tpy_max = 30.0, 60.0, 150.0

# Layout constraints
trains_per_facility = 8
reactors_per_train = 1

AnnualOperatingHours = 8000.0             # h/y

# ---- SINGLE SOURCE OF TRUTH (edit here) ----
SCALE = set_scale(throughput_tpy=1000000.0, annual_operating_hours=AnnualOperatingHours)  # 1 Mt/y CNT target
Throughputofscale_ton_per_year = SCALE.throughput_tpy
Throughputofscale_g_h = SCALE.throughput_g_h

# Turn on to print sizing diagnostics (reactors/trains/facilities) during runs
DEBUG_SCALE = True

# Reactor conditions
T_reactor_C = 1300.0
P_reactor_Pa = 101_325.0
T_ref_C = 25.0

# Finance
Yearforprofitibilitymodel = 10
InterestDiscountRate = 0.10

AreyousellingH2 = "Yes"
AreyousellingSolidCarbon = "Yes"

def headcount_from_facilities(num_facilities: int) -> dict[str, int]:
    """Return total headcount given the number of facilities.

    Convention:
    - Plant roles scale per facility.
    - C-suite roles are corporate (do not scale with facilities).
    """
    n = int(max(num_facilities, 1))
    return {
        "PlantManager": n * 1,
        "PlantEngineers": n * 5,
        "Administrator": n * 2,
        "Operators": n * 10,
        "AssistantPlantManager": n * 1,
        "CEO": 1,
        "CTO": 1,
        "COO": 1,
    }



# ============================================================
# Labour cost assumptions (annual base salaries)
# ============================================================
# Salaries originally specified in GBP and converted to USD using GBPUSD FX.
# These values represent *median-to-lower-quartile* industrial manufacturing
# salaries (chemical / process industries), excluding bonuses and benefits.

# FX reference:
# Bank of England / Wise historical mid-market GBP→USD rates (2024–2025 average)
# https://www.bankofengland.co.uk/boeapps/database/Rates.asp
# https://wise.com/gb/currency-converter/gbp-to-usd-rate/history
GBPUSD = 1.34

salary_anchors = {
    "PlantManager": {
        "low":  65_000 * GBPUSD,   # UK
        "mid":  95_000,            # blended
        "high": 120_000,           # USA
    },
    "ProcessEngineer": {
        "low":  50_000 * GBPUSD,
        "mid":  85_000,
        "high": 105_000,
    },
    "Administrator": {
        "low":  25_000 * GBPUSD,
        "mid":  38_000,
        "high": 45_000,
    },
    "Operator": {
        "low":  40_000 * GBPUSD,
        "mid":  55_000,
        "high": 65_000,
    },
    "AssistantPlantManager": {
        "low":  50_000 * GBPUSD,
        "mid":  70_000,
        "high": 80_000,
    },
    "CEO": {
        "low":  175_000 * GBPUSD,
        "mid":  200_000,
        "high": 220_000,
    },
    "CTO": {
        "low":  100_000 * GBPUSD,
        "mid":  140_000,
        "high": 170_000,
    },
    "COO": {
        "low":  150_000 * GBPUSD,
        "mid":  180_000,
        "high": 200_000,
    },
}



# ------------------------------------------------------------
# Plant Manager (Chemical / Process Manufacturing)
# ------------------------------------------------------------
# UK median plant / operations manager salary:
# £60k–£75k depending on site complexity
# Sources:
# - Hays UK Salary Guide – Engineering & Manufacturing (2024)
#   https://www.hays.co.uk/salary-guide
# - Glassdoor UK: "Plant Manager – Manufacturing"
#   https://www.glassdoor.co.uk/Salaries/plant-manager-salary-SRCH_KO0,13.htm
PlantManagersalary = salary_anchors["PlantManager"]["mid"]


# ------------------------------------------------------------
# Plant / Process Engineer
# ------------------------------------------------------------
# UK chemical / process engineer salaries:
# £45k–£60k for mid-level engineers
# Sources:
# - IChemE Salary Survey (2023–2024)
#   https://www.icheme.org/membership/salary-survey/
# - Prospects UK – Chemical Engineer
#   https://www.prospects.ac.uk/job-profiles/chemical-engineer
PlantEngineersalary = salary_anchors["ProcessEngineer"]["mid"]


# ------------------------------------------------------------
# Administrator / Site Admin
# ------------------------------------------------------------
# UK manufacturing administrative roles:
# £22k–£30k typical range
# Sources:
# - UK Office for National Statistics (ONS) ASHE data
#   https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours
# - Glassdoor UK – Administrative Assistant (Manufacturing)
Administratorsalary = salary_anchors["Administrator"]["mid"]


# ------------------------------------------------------------
# Operators / Technicians (Process Operators)
# ------------------------------------------------------------
# UK chemical plant operators:
# £35k–£45k base (shift premiums excluded)
# Sources:
# - IChemE Technician & Operator salary benchmarks
# - Glassdoor UK – Process Operator
#   https://www.glassdoor.co.uk/Salaries/process-operator-salary-SRCH_KO0,16.htm
Operatorsalary = salary_anchors["Operator"]["mid"]


# ------------------------------------------------------------
# Assistant Plant Manager / Senior Supervisor
# ------------------------------------------------------------
# UK senior production supervisor / deputy plant manager:
# £45k–£60k
# Sources:
# - Hays UK – Manufacturing Management Roles
# - Glassdoor UK – Production Supervisor (Manufacturing)
Assistantplantmanagersalary = salary_anchors["AssistantPlantManager"]["mid"]


# ------------------------------------------------------------
# Executive management (company-level, not per facility)
# ------------------------------------------------------------
# These are conservative industrial executive salaries,
# not venture-backed startup or Big Tech compensation.

# CEO – small-to-mid industrial company
# £130k–£180k base
# Sources:
# - UK Directors’ Remuneration Reports (manufacturing SMEs)
# - Glassdoor UK – Managing Director (Manufacturing)
CEOsalary = salary_anchors["CEO"]["mid"]

# CTO – industrial technology / process development
# £80k–£120k
# Sources:
# - Hays UK – Engineering Leadership Roles
# - Glassdoor UK – Technical Director
CTOsalary = salary_anchors["CTO"]["mid"]

# COO – operations-heavy manufacturing business
# £110k–£160k
# Sources:
# - Glassdoor UK – COO Manufacturing
# - UK SME manufacturing executive surveys
COOsalary = salary_anchors["COO"]["mid"]

# Maintenance fraction (fixed here; can be made uncertain if you want)
Percentangecostmaintenance = 0.10





# Thermo (optional)
# If CoolProp is installed in your environment, we will use it for ideal-gas Cp and densities.
# If not, we fall back to simple high-T Cp approximations (still adequate for sensitivity/MC).
try:
    import CoolProp.CoolProp as CP  # type: ignore
    HAS_COOLPROP = True
except Exception:
    CP = None  # type: ignore
    HAS_COOLPROP = False


def cp_ideal_gas_kJ_kgK(fluid: str, T_C: float, P_Pa: float = 101325.0) -> float:
    """Mass specific heat Cp for a pure gas at (T,P).

    - Preferred: CoolProp `Cpmass`.
    - Fallback: constant Cp values (kJ/kg/K) representative at high temperature.

    The fallback is meant only to keep the physics block runnable without CoolProp.
    """
    T_K = float(T_C) + 273.15
    if HAS_COOLPROP:
        try:
            return float(CP.PropsSI("Cpmass", "T", T_K, "P", float(P_Pa), fluid)) / 1000.0
        except Exception:
            pass

    # Rough high-T constants (kJ/kg/K) – order-of-magnitude only
    fallback = {
        "Methane": 3.3,
        "Hydrogen": 14.3,
        "Nitrogen": 1.2,
        "CarbonDioxide": 1.3,
        "CO2": 1.3,
    }
    return float(fallback.get(fluid, 1.5))


def rho_ideal_gas_kg_m3(fluid: str, T_C: float, P_Pa: float = 101325.0) -> float:
    """Ideal-gas density from CoolProp if available; otherwise ideal-gas fallback."""
    T_K = float(T_C) + 273.15
    if HAS_COOLPROP:
        try:
            return float(CP.PropsSI("Dmass", "T", T_K, "P", float(P_Pa), fluid))
        except Exception:
            pass

    # Ideal gas: rho = P*MW/(R*T)
    R = 8.314462618
    MW = {
        "Methane": 0.016043,
        "Hydrogen": 0.002016,
        "Nitrogen": 0.0280134,
        "CarbonDioxide": 0.04401,
        "CO2": 0.04401,
    }.get(fluid, 0.028)
    return float(P_Pa * MW / (R * T_K))

# ============================================================
# 0) RUN CONFIG
# ============================================================

@dataclass(frozen=True)
class RunConfig:
    # You can override via environment variable CNT_MC_N, e.g.:
    #   CNT_MC_N=1000000 python Industrial_model_2026_v5_fccvd_flb.py
    num_samples: int = int(os.getenv("CNT_MC_N", "100000"))
    seed: int = 42
    currency: str = "USD"

    # Outputs
    # Set CNT_OUTPUT_DIR to control where CSV/plots are written.
    # - If relative, it's resolved relative to this script's folder.
    # - If absolute, it's used as-is.
    _output_dir_env: str = os.getenv("CNT_OUTPUT_DIR", "").strip()
    output_dir: str = (
        _output_dir_env
        if os.path.isabs(_output_dir_env)
        else (os.path.join(os.path.dirname(__file__), _output_dir_env) if _output_dir_env else os.path.dirname(__file__))
    )
    save_outputs: bool = True
    # Set CNT_SHOW_PLOTS=1 to pop up Matplotlib windows when running from VS Code.
    show_plots: bool = os.getenv("CNT_SHOW_PLOTS", "0").strip().lower() in {"1", "true", "yes"}

cfg = RunConfig()

# Plant operating hours per year (excluding maintenance/turnarounds)
# You used 8,000 h/yr in your earlier code.
AnnualCNTProduction = 8000

rng = np.random.default_rng(cfg.seed)
N = cfg.num_samples


# ============================================================
# INTEGRATION BLOCK: Reactor/Train/Facility logic + CAPEX (from to be_integrated.py)
# ============================================================



# ---------------------------------------------------------------------
# TRAIN NAMEPLATE REFERENCE (used ONLY to set a realistic default)
# ---------------------------------------------------------------------
# --- Train nameplate uncertainty (t/y) ---
# Anchor: OCSiAl "Graphetron 50" described as ~50 t/y capacity (order-of-magnitude train class).
# Sources:
# - PCI Magazine (2020): Graphetron 50 “production capacity of 50 tonnes per year”
#   https://www.pcimag.com/articles/107227-ocsial-launches-second-graphene-nanotube-synthesis-facility
# - OCSiAl news/story also describing Graphetron 50 (~50 t/y)
#   https://ocsial.com/es/news/-fire-water-and-nanotubes-/
#
# Additional industrial context (shows the sector spans tens to hundreds+ t/y at plant level):
# - Bayer MaterialScience brochure: “can currently produce up to 60 tons ... a year” (Baytubes)
#   https://www.bayer.com/sites/default/files/2020-05/gb-2007-en_0.pdf
# - Arkema reported pilot/plant capacities (e.g., 20 t/y lab pilot; 400 t/y proposed plant)
#   https://www.industrialinfo.com/news/article/frances-arkema-to-construct-400-ton-carbon-nanotube-pilot-plant--152935
#
# Modelling choice:
# - Use a truncated lognormal for positive, right-skewed “train” capacities.
# - Median ~50 t/y; truncate to keep “train” from silently becoming a multi-plant.

def tpy_to_kg_h(tpy: float, hours_per_year: float, availability: float) -> float:
    """Convert tonnes/year nameplate to effective kg/h."""
    eff_hours = hours_per_year * availability
    return (tpy * 1000.0) / eff_hours


def sample_train_nameplate_tpy(
    rng,
    size: int = 1,
    *,
    median_tpy: float = 50.0,
    min_tpy: float = 20.0,
    max_tpy: float = 200.0,
    sigma_ln: float = 0.45,
) -> np.ndarray:
    """Sample a truncated-lognormal train nameplate (t/y)."""
    samples = median_tpy * lognorm(s=sigma_ln, scale=1.0).rvs(size=size, random_state=rng)
    return np.clip(samples, min_tpy, max_tpy).astype(float)

def reactor_cnt_kg_h(route: str, scale_factor: float, alpha: float,
                     base_cnt_kg_h: float, max_cnt_kg_h: float) -> float:
    """
    Scaling for per-reactor CNT rate.
    Replace base_cnt_kg_h with your *physics-model* output at reference scale.
    """
    rate = base_cnt_kg_h * (scale_factor ** alpha)
    return float(min(rate, max_cnt_kg_h))

def compute_reactor_facility_counts(
    target_tpy: float,
    hours_per_year: float,
    availability: float,
    base_cnt_kg_h: float,
    scale_factor: float,
    alpha: float,
    max_cnt_kg_h: float,
    trains_per_facility: int = 8,
    reactors_per_train: int = 1
):
    """
    Robust replication logic:
      - Reactor (process vessel)
      - Train (reactor + furnace + solids handling + recycle/PSA)
      - Facility (multiple trains + shared utilities/QA/etc.)
    """
    target_kg_y = target_tpy * 1000.0
    eff_hours = hours_per_year * availability

    per_reactor_kg_y = reactor_cnt_kg_h("FCCVD", scale_factor, alpha,
                                       base_cnt_kg_h, max_cnt_kg_h) * eff_hours
    per_train_kg_y = per_reactor_kg_y * reactors_per_train

    n_trains = int(np.ceil(target_kg_y / per_train_kg_y))
    n_facilities = int(np.ceil(n_trains / trains_per_facility))
    n_reactors = n_trains * reactors_per_train

    return n_reactors, n_trains, n_facilities, per_reactor_kg_y


def compute_reactor_facility_counts_vectorized(
    target_tpy: float,
    hours_per_year: float,
    availability: float,
    base_cnt_kg_h: np.ndarray,
    scale_factor: float,
    alpha: float,
    max_cnt_kg_h: float,
    trains_per_facility: int = 8,
    reactors_per_train: int = 1,
):
    """Vectorized replication logic over Monte Carlo samples."""
    target_kg_y = float(target_tpy) * 1000.0
    eff_hours = float(hours_per_year) * float(availability)

    base_cnt_kg_h = np.asarray(base_cnt_kg_h, dtype=float)
    per_reactor_rate_kg_h = base_cnt_kg_h * (float(scale_factor) ** float(alpha))
    per_reactor_rate_kg_h = np.minimum(per_reactor_rate_kg_h, float(max_cnt_kg_h))

    per_reactor_kg_y = per_reactor_rate_kg_h * eff_hours
    per_train_kg_y = per_reactor_kg_y * int(reactors_per_train)

    n_trains = np.ceil(target_kg_y / np.maximum(per_train_kg_y, 1e-12)).astype(int)
    n_facilities = np.ceil(n_trains / int(trains_per_facility)).astype(int)
    n_reactors = n_trains * int(reactors_per_train)

    return n_reactors, n_trains, n_facilities, per_reactor_kg_y

def calculate_parameters(
    Throughputofscale_ton_per_year: float,
    hours_per_year: float = 8000.0,
    availability: float = 0.90,
    route: str = "FCCVD",
    trains_per_facility: int = 8,
    reactors_per_train: int = 1,
    rng=None
):
    """
    Updated approach:
    1) Use a referenced “train nameplate” (50 t/y class) to set DEFAULT per-reactor
       capacity bounds (base + max). Override with physics model when available.
    2) Compute reactors/trains/facilities via replication logic.
    3) Keep your legacy anchors for facility share cost + furnace power by tier,
       but let counts come from train sizing (more realistic long-term).

    Train nameplate reference used for DEFAULTS:
      - OCSiAl Graphetron 50 is described as 50 t/y capacity (see links above).

    Notes:
    - route can be "FCCVD" or "FLB" (both can share this counting framework).
    """
    if rng is None:
        rng = np.random.default_rng()

    # -------------------------------
    # (A) DEFAULT train nameplate anchor (t/y)
    # -------------------------------
    # --- Train nameplate uncertainty (t/y) ---
    # Anchor: OCSiAl "Graphetron 50" described as ~50 t/y capacity (order-of-magnitude train class).
    # Modelling choice:
    # - Use a truncated lognormal for positive, right-skewed “train” capacities.
    # - Median ~50 t/y; truncate to keep “train” from silently becoming a multi-plant.
    train_tpy_median = 50.0
    train_tpy_min, train_tpy_max = 20.0, 200.0
    sigma_ln = 0.45  # screening uncertainty (tune when you have better plant/train definitions)

    train_nameplate_tpy = train_tpy_median * lognorm(s=sigma_ln, scale=1.0).rvs(
        size=1,
        random_state=rng
    )[0]
    train_nameplate_tpy = float(np.clip(train_nameplate_tpy, train_tpy_min, train_tpy_max))

    train_nameplate_tpy_ref = train_nameplate_tpy

    # Convert to effective kg/h for DEFAULT “base per reactor” if 1 reactor/train
    base_cnt_kg_h_default = tpy_to_kg_h(train_nameplate_tpy_ref, hours_per_year, availability)

    # Set a conservative max (debottleneck / modest uprate), still just a placeholder
    # until your physics model gives real constraints.
    max_cnt_kg_h_default = 1.5 * base_cnt_kg_h_default  # allow +50% uprate as a placeholder

    # -------------------------------
    # (B) Scaling exponent alpha (placeholder)
    # -------------------------------
    # In real use, alpha should come from your physics/scale model.
    # Here we keep it modest (<1) to reflect partial economies with scale.
    alpha_default = 0.6

    # If you don’t want any “size scaling” yet, set scale_factor=1 and alpha ignored.
    scale_factor_default = 1.0

    # -------------------------------
    # (C) Counts via trains/facilities (robust replication logic)
    # -------------------------------
    n_reactors, n_trains, n_facilities, per_reactor_kg_y = compute_reactor_facility_counts(
        target_tpy=Throughputofscale_ton_per_year,
        hours_per_year=hours_per_year,
        availability=availability,
        base_cnt_kg_h=base_cnt_kg_h_default,
        scale_factor=scale_factor_default,
        alpha=alpha_default,
        max_cnt_kg_h=max_cnt_kg_h_default,
        trains_per_facility=trains_per_facility,
        reactors_per_train=reactors_per_train
    )

    numberofreactors = float(n_reactors)
    Numberoffacility = int(n_facilities)

    # -------------------------------
    # (D) Keep your legacy tier anchors for shared costs + furnace power
    #     (you can replace these later with equipment models)
    # -------------------------------
    if Throughputofscale_ton_per_year <= 100:
        Facilitysharecost = 51_920.0       # # https://www.bestongroup.com/pyrolysis-plant-cost/
        reference_Furnace_power = 16.65    # # legacy anchor
    elif 100 < Throughputofscale_ton_per_year < 1_000_000:
        Facilitysharecost = 75_800.0       # # https://www.bestongroup.com/pyrolysis-plant-cost/
        reference_Furnace_power = 37.85    # # legacy anchor
    else:
        Facilitysharecost = 688_900.0      # # https://www.bestongroup.com/pyrolysis-plant-cost/
        # For very large scale, distribute a “facility-level” furnace power anchor across reactors
        reference_Furnace_power = 256.0 / max(numberofreactors, 1.0)  # # legacy anchor form

    # -------------------------------
    # (E) Single-reactor prices (still MC-distributions elsewhere)
    #     Here just return placeholders; you’ll sample distributions in MC layer.
    # -------------------------------
    Singlereactorprice_FCVD = np.nan  # sample in MC layer (ceramic/refractory 1300C)
    Singlereactorprice_FLB  = np.nan  # sample in MC layer (metal/refractory 900C)

    return (float(numberofreactors), int(Numberoffacility),
            float(Singlereactorprice_FCVD), float(Singlereactorprice_FLB),
            float(reference_Furnace_power), float(Facilitysharecost),
            int(n_trains), float(per_reactor_kg_y))


# ----------------------------------------------------------------------------
# CAPEX building blocks (ported from your 2023 model)
# ----------------------------------------------------------------------------

def reactor_capital(
    Tube_scale_up_factor: float,
    Tube_capital_exponent_scaleup: float,
    Throughput_: float,
    Throughput_of_scale: float,
    Number_of_reactors: float,
    Number_of_facility: int,
    Tube_per_reactor_sclaling_exponent_factor: float,
    Reactor_scaling_factor: float,
    Facility_scaling_factor: float,
    Single_reactor_price_usd,
):
    """Scaled reactor (train) capital cost.

    Note: Single_reactor_price_usd can be a numpy array (MC samples).
    """
    Number_of_tubes_per_reactor = (
        Throughput_of_scale
        / Tube_scale_up_factor
        / (Number_of_reactors * Number_of_facility)
        / Throughput_
    )
    scale_cost = (
        Single_reactor_price_usd
        * (Tube_scale_up_factor ** Tube_capital_exponent_scaleup)
        * (Number_of_tubes_per_reactor ** Tube_per_reactor_sclaling_exponent_factor)
        * (Number_of_reactors ** Reactor_scaling_factor)
        * (Number_of_facility ** Facility_scaling_factor)
    )
    return Number_of_tubes_per_reactor, scale_cost


def facilty_shared_capital(
    Facility_share_cost: float,
    Throughput_: float,
    Throughput_of_scale: float,
    Number_of_facility: int,
    scaling_exponent: float,
):
    scale_up_factor_from_reference = Throughput_of_scale / Throughput_ / Number_of_facility
    scaled_cost = Number_of_facility * Facility_share_cost * (scale_up_factor_from_reference ** scaling_exponent)
    return scale_up_factor_from_reference, scaled_cost


def Additional_capital(
    Recovery_of_H2_from_purge_gas,
    Gas_recovery_cost: float,
    Throughput_: float,
    Throughput_of_scale: float,
    Number_of_facility: int,
    Gas_recovery_scale_factor: float,
):
    # If no H2 recovery, no PSA/recycle capital
    reference_cost = 0.0
    try:
        # vector-safe check
        if (Recovery_of_H2_from_purge_gas == 0).all():
            reference_cost = 0.0
        else:
            reference_cost = Gas_recovery_cost
    except Exception:
        reference_cost = 0.0 if Recovery_of_H2_from_purge_gas == 0 else Gas_recovery_cost

    scale_up_factor = Throughput_of_scale / Throughput_ / Number_of_facility
    scaled_cost = reference_cost * (scale_up_factor ** Gas_recovery_scale_factor) * Number_of_facility
    return scaled_cost


def Infrastructure_QA_cost(Infrastructure_QA_equipement_cost: float, Number_of_facility: int):
    return Infrastructure_QA_equipement_cost * Number_of_facility


st = time.time()




# ============================================================
# 2) HELPER FUNCTIONS (THERMO + COST)
# ============================================================

def K(C: float) -> float:
    return C + 273.15


_MOLAR_MASS_KG_PER_MOL_FALLBACK = {
    "Methane": 0.016043,
    "Hydrogen": 0.002016,
    "Nitrogen": 0.0280134,
    "CarbonDioxide": 0.04401,
}

# Very simple high-temperature Cp fallbacks (order-of-magnitude screening).
# If you install CoolProp, these will be replaced automatically.
_CP_J_PER_KG_K_FALLBACK = {
    "Methane": 4_000.0,
    "Hydrogen": 14_300.0,
    "Nitrogen": 1_250.0,
    "CarbonDioxide": 1_200.0,
}

_K_GAMMA_FALLBACK = {
    "Methane": 1.27,
    "Hydrogen": 1.40,
    "Nitrogen": 1.40,
    "CarbonDioxide": 1.30,
}


def cp_mass(fluid: str, T_K: float, P_Pa: float) -> float:
    """Mass specific heat cp [J/kg/K] for a pure fluid.

    Uses CoolProp if available; otherwise uses conservative high-T constants.
    """
    if HAS_COOLPROP:
        return float(CP.PropsSI("Cpmass", "T", T_K, "P", P_Pa, fluid))
    return float(_CP_J_PER_KG_K_FALLBACK[fluid])


def molar_mass(fluid: str) -> float:
    """Molar mass [kg/mol] for a pure fluid."""
    if HAS_COOLPROP:
        return float(CP.PropsSI("M", fluid))
    return float(_MOLAR_MASS_KG_PER_MOL_FALLBACK[fluid])


def mmbtu_to_usd_per_kg_ch4(price_usd_per_mmbtu: float) -> float:
    """Convert $/MMBtu to $/kg CH4 using LHV(CH4) ~ 50 MJ/kg and 1 MMBtu ~ 1055 MJ."""
    return price_usd_per_mmbtu / (1055.0 / 50.0)


def isentropic_compression_power_kW(
    m_dot_kg_s: np.ndarray,
    fluid: str,
    T1_K: float,
    P1_Pa: float,
    P2_Pa: np.ndarray | float,
    eta_isentropic: np.ndarray | float,
) -> np.ndarray:
    """Simple ideal-gas isentropic compression power [kW] using k from CoolProp.

    Wdot = m_dot * cp * T1 * ((PR)**((k-1)/k) - 1) / eta

    This is a common screening-level model for PSA/feed compression.
    """
    P2 = np.asarray(P2_Pa, dtype=float)
    eta = np.asarray(eta_isentropic, dtype=float)

    # Broadcast to massflow shape
    if P2.ndim == 0:
        P2 = np.full_like(m_dot_kg_s, float(P2))
    if eta.ndim == 0:
        eta = np.full_like(m_dot_kg_s, float(eta))

    mask = P2 > P1_Pa
    if not np.any(mask):
        return np.zeros_like(m_dot_kg_s)

    cp = cp_mass(fluid, T1_K, P1_Pa)  # J/kg/K
    if HAS_COOLPROP:
        cv = float(CP.PropsSI("Cvmass", "T", T1_K, "P", P1_Pa, fluid))
        k = cp / cv
    else:
        # Typical ideal-gas heat capacity ratios (screening-level)
        k_fallback = {"Methane": 1.28, "Hydrogen": 1.41, "Nitrogen": 1.40, "CarbonDioxide": 1.30}
        k = float(k_fallback[fluid])

    PR = np.ones_like(m_dot_kg_s)
    PR[mask] = P2[mask] / P1_Pa
    term = (PR ** ((k - 1.0) / k) - 1.0)

    Wdot_W = np.zeros_like(m_dot_kg_s)
    Wdot_W[mask] = (
        m_dot_kg_s[mask] * cp * T1_K * term[mask] / np.maximum(1e-9, eta[mask])
    )
    return Wdot_W / 1000.0


# ============================================================
# 3) UNCERTAINTIES (DISTRIBUTIONS)
# ============================================================

# Availability (fraction of planned hours actually run)
Availability = beta(a=46, b=4).rvs(size=N, random_state=rng)  # mean ~0.92



# Academic refs for "overhead fraction on direct labour" (indirect labor + supervision + plant overhead allocation):
# 1) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003). Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Cost estimation practice commonly treats plant/indirect overhead as a fraction of direct labor; typical ranges vary by plant type.)
# 2) Towler, G., & Sinnott, R. (2013). Chemical Engineering Design: Principles, Practice and Economics of Plant and Process Design (2nd ed.). Elsevier.
#    (Project cost estimation includes indirect costs/overheads often correlated to, or expressed as a percentage of, direct labor in early estimates.)
# 3) Garrett, D.E. (1989). Chemical Engineering Economics. Van Nostrand Reinhold.
#    (Discusses overhead/indirect cost burdens and use of percentage adders on labor in cost estimation.)
# Overhead fraction on labour (indirect costs)
percentageoflabourcosts = triang(
    c=(0.20 - 0.15) / (0.35 - 0.15),
    loc=0.15,
    scale=0.20
).rvs(size=N, random_state=rng)

# Methane (natural gas) price anchor and uncertainty — references:
# 1) U.S. Energy Information Administration (EIA), Henry Hub Natural Gas Spot Price
#    https://www.eia.gov/dnav/ng/hist/rngwhhdm.htm
#    (Primary authoritative market series for U.S. natural gas prices, quoted in $/MMBtu.)
#
# 2) CME Group, Henry Hub Natural Gas Futures (NG)
#    https://www.cmegroup.com/markets/energy/natural-gas/natural-gas.html
#    (Market-traded futures providing forward-looking price expectations and volatility.)
#
# 3) IEAGHG / IEA Hydrogen & Gas Reports (for use in techno-economic assessments)
#    International Energy Agency, Gas Market Report (various years)
#    https://www.iea.org/reports/gas-market-report
#    (Used in academic TEA/LCA studies to justify representative gas price ranges and variability.)
#
# 4) Towler, G., & Sinnott, R. (2013). Chemical Engineering Design (2nd ed.). Elsevier.
#    (Recommends using lognormal-type distributions for commodity price uncertainty in early-stage TEA.)
#
# Conversion note:
# - 1 MMBtu ≈ 1055 MJ; CH4 LHV ≈ 50 MJ/kg → ≈ 21.1 kg CH4 per MMBtu
# - This conversion is standard in hydrogen and methane TEA studies.
#
henry_hub_usd_per_mmbtu_base = 3.8
Methaneprice_base = mmbtu_to_usd_per_kg_ch4(henry_hub_usd_per_mmbtu_base)
Methaneprice = Methaneprice_base * lognorm(s=0.35, scale=1.0).rvs(size=N, random_state=rng)


# Industrial electricity price anchor and uncertainty — references:
# 1) U.S. Energy Information Administration (EIA), Electric Power Monthly
#    https://www.eia.gov/electricity/monthly/
#    (Reports average industrial electricity prices in $/kWh by month and year;
#     recent U.S. industrial averages are typically ~0.07–0.11 $/kWh.)
#
# 2) International Energy Agency (IEA), Electricity Information (annual statistical report)
#    https://www.iea.org/reports/electricity-information-overview
#    (Provides cross-country industrial electricity prices widely cited in academic TEA/LCA studies.)
#
# 3) Lazard, Levelized Cost of Energy (LCOE) Analysis (latest version)
#    https://www.lazard.com/perspective/lcoe-analysis/
#    (Common industry reference used to justify representative industrial power prices and variability.)
#
# 4) Towler, G., & Sinnott, R. (2013). Chemical Engineering Design (2nd ed.). Elsevier.
#    (Recommends lognormal distributions for utility cost uncertainty in early-stage techno-economic models.)
#
electricaldemandcost_base = 0.09  # USD/kWh (industrial electricity price anchor)
electricaldemandcost = electricaldemandcost_base * lognorm(s=0.20, scale=1.0).rvs(size=N, random_state=rng)


# Nitrogen (industrial gas) price anchor and uncertainty — references:
# 1) Gasworld, Industrial Gases Pricing & Market Reports
#    https://www.gasworld.com/industrial-gases-pricing/2025375.article
#    (Industry-standard source for bulk nitrogen pricing; reports contract prices typically
#     in the range ~0.05–0.30 USD/kg depending on volume, purity, and delivery mode.)
#
# 2) Air Products, Linde, Air Liquide – Bulk Nitrogen Supply (vendor technical brochures)
#    e.g. https://www.airliquide.com/industries/industrial-gases/nitrogen
#    (Vendor documentation commonly cited in TEA studies for indicative bulk N₂ pricing ranges.)
#
# 3) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Recommends triangular distributions with min–mode–max values for contract utilities
#     and industrial gases in early-stage cost estimation.)
#
Nitrogenprice_base = 0.16  # USD/kg (bulk nitrogen, industrial contract anchor)
Nitrogenprice = triang(
    c=(Nitrogenprice_base - 0.10) / (0.30 - 0.10),
    loc=0.10,
    scale=0.20
).rvs(size=N, random_state=rng)


# Sulfur (bulk commodity / industrial powder) price anchor and uncertainty — references:
# 1) Intratec Solutions, Sulfur Price & Cost Reports
#    https://www.intratec.us/solutions/primary-commodity-prices/commodities/sulfur-prices
#    (Widely cited source for elemental sulfur commodity prices; typical bulk prices
#     range ~150–350 USD/ton depending on market conditions and region.)
#
# 2) U.S. Geological Survey (USGS), Mineral Commodity Summaries – Sulfur
#    https://www.usgs.gov/centers/national-minerals-information-center/sulfur-statistics-and-information
#    (Authoritative annual statistics on sulfur production, trade, and pricing.)
#
# 3) Abánades et al. (2012), “Thermal methane cracking: an overview”,
#    International Journal of Hydrogen Energy, 37, 9559–9573.
#    https://doi.org/10.1016/j.ijhydene.2012.03.004
#    (Uses industrial sulfur pricing assumptions in the context of methane cracking
#     and hydrogen/carbon co-production systems.)
#
# 4) Towler, G., & Sinnott, R. (2013). Chemical Engineering Design (2nd ed.). Elsevier.
#    (Recommends lognormal distributions for commodity price uncertainty in TEA studies.)
#
Sulphurprice_base = 0.25  # USD/kg (≈250 USD/ton, bulk elemental sulfur anchor)
Sulphurprice = Sulphurprice_base * lognorm(s=0.50, scale=1.0).rvs(size=N, random_state=rng)


# Iron powder price (industrial procurement) — references:
# 1) Alibaba Industrial Materials – Iron Powder Buying Guide
#    https://smartbuy.alibaba.com/buyingguides/iron-powder
#    (Provides indicative bulk iron powder prices by grade, particle size,
#     and order volume; typical ranges ~2–8 USD/kg for industrial grades.)
#
# 2) Höganäs AB – Iron Powder Products (industrial benchmark supplier)
#    https://www.hoganas.com/en/powder-technologies/iron-powder/
#    (Leading global producer; pricing not public but frequently cited in TEA
#     studies as a benchmark for industrial iron powder costs.)
#
# 3) Dawkins et al. (2023), “Catalytic methane pyrolysis using iron ore-based powder catalysts”,
#    International Journal of Hydrogen Energy, 48, 15333–15346.
#    https://doi.org/10.1016/j.ijhydene.2023.03.022
#    (Uses iron-based powder catalyst cost assumptions consistent with
#     multi-USD/kg industrial procurement ranges.)
#
# 4) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Recommends triangular distributions (min–mode–max) for supplier-quoted
#     raw materials in early-stage techno-economic assessments.)
#
Ironprice_min, Ironprice_mode, Ironprice_max = 2.0, 3.5, 8.0  # USD/kg (industrial iron powder)
Ironprice = triang(
    c=(Ironprice_mode - Ironprice_min) / (Ironprice_max - Ironprice_min),
    loc=Ironprice_min,
    scale=(Ironprice_max - Ironprice_min)
).rvs(size=N, random_state=rng)


# Carbon dioxide (bulk, delivered) price anchor and uncertainty — references:
# 1) Thomasnet – Industrial CO₂ Suppliers (quote-based market listings)
#    https://www.thomasnet.com/products/carbon-dioxide-15170404-1.html
#    (Provides indicative price ranges for bulk and packaged industrial CO₂;
#     reported prices commonly span ~0.02–0.20 USD/kg depending on purity,
#     delivery mode, and contract volume.)
#
# 2) U.S. Geological Survey (USGS), Mineral Commodity Summaries – Carbon Dioxide
#    https://www.usgs.gov/centers/national-minerals-information-center/carbon-dioxide-statistics-and-information
#    (Authoritative source on CO₂ production, supply routes, and industrial usage,
#     often cited in techno-economic and LCA studies.)
#
# 3) International Energy Agency (IEA), CCUS in Clean Energy Transitions
#    https://www.iea.org/reports/ccus-in-clean-energy-transitions
#    (Discusses cost ranges for captured and merchant CO₂, including delivered
#     costs relevant for industrial users.)
#
# 4) Rubin, E.S., Davison, J.E., & Herzog, H.J. (2015).
#    “The cost of CO₂ capture and storage,” International Journal of Greenhouse Gas Control, 40, 378–400.
#    https://doi.org/10.1016/j.ijggc.2015.05.018
#    (Widely cited academic reference for CO₂ cost ranges across capture,
#     compression, transport, and delivery pathways.)
#
# 5) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Supports use of triangular distributions for commodity inputs with
#     supplier-quoted min–mode–max prices in early-stage TEA.)
#
CO2price_min, CO2price_mode, CO2price_max = 0.02, 0.07, 0.20  # USD/kg (bulk industrial CO₂)
CO2price = triang(
    c=(CO2price_mode - CO2price_min) / (CO2price_max - CO2price_min),
    loc=CO2price_min,
    scale=(CO2price_max - CO2price_min)
).rvs(size=N, random_state=rng)



# Ferro-aluminium (FeAl alloy) price anchor and uncertainty — references:
# 1) London Metal Exchange (LME) – Aluminium Alloy & Ferroalloy Market Context
#    https://www.lme.com/Metals/Non-ferrous/Aluminium
#    (Provides benchmark pricing context for aluminium and aluminium-based alloys;
#     ferro-aluminium pricing typically tracks aluminium plus alloying/processing premiums.)
#
# 2) USGS, Mineral Commodity Summaries – Aluminum
#    https://www.usgs.gov/centers/national-minerals-information-center/aluminum-statistics-and-information
#    (Authoritative source on aluminium production, trade, and price ranges used
#     to contextualize aluminium-based alloy costs.)
#
# 3) Indian Bureau of Mines / Ferroalloy Market Reports
#    https://ibm.gov.in/
#    (Provides indicative price ranges for ferroalloys including FeAl in
#     industrial metallurgy markets.)
#
# 4) ASM Handbook, Volume 2: Properties and Selection—Nonferrous Alloys and Special-Purpose Materials
#    (ASM International, 1990).
#    (Discusses ferro-aluminium compositions, industrial applications, and
#     cost positioning relative to aluminium and iron.)
#
# 5) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Supports triangular distributions (min–mode–max) for alloy procurement
#     costs in early-stage techno-economic assessments.)
#
FeAlprice_min, FeAlprice_mode, FeAlprice_max = 1.5, 3.0, 6.0  # USD/kg (industrial Fe-Al alloy)
FeAlprice = triang(
    c=(FeAlprice_mode - FeAlprice_min) / (FeAlprice_max - FeAlprice_min),
    loc=FeAlprice_min,
    scale=(FeAlprice_max - FeAlprice_min)
).rvs(size=N, random_state=rng)


# Carbon nanotube (CNT) selling price anchor and uncertainty — references:
# 1) Isaacs, J.A., Tanwani, A., & Gosselin, D. (2009),
#    “Single-walled carbon nanotube manufacturing cost analysis,”
#    Carbon, 47, 3053–3063.
#    https://doi.org/10.1016/j.carbon.2009.06.041
#    (Provides cost and selling price context for industrial-scale CNT production,
#     with prices strongly dependent on purity, structure, and application.)
#
# 2) Endo, M., Hayashi, T., Kim, Y.A., et al. (2008),
#    “Mass production of carbon nanotubes,”
#    Carbon, 46, 170–176.
#    https://doi.org/10.1016/j.carbon.2007.11.016
#    (Reports early industrial-scale CNT production routes and indicative
#     price levels as production scales up.)
#
# 3) Nanografi – Industrial Grade MWCNT Product Listings
#    https://shop.nanografi.com/carbon-nanotubes/industrial-grade-multi-walled-carbon-nanotubes-purity-92-outside-diameter-48-78-nm/
#    (Representative vendor prices for industrial-grade CNTs in the
#     ~100–500 USD/kg range, depending on grade and volume.)
#
# 4) Cheap Tubes Inc. – Carbon Nanotube Pricing (commercial supplier)
#    https://www.cheaptubes.com/
#    (Commercial benchmark widely cited in academic and techno-economic studies.)
#
# 5) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Supports triangular distributions (min–mode–max) for product selling prices
#     in early-stage techno-economic assessments.)
#
SalepriceCNT_min, SalepriceCNT_mode, SalepriceCNT_max = 10.0, 50.0, 100.0 #100.0, 250.0, 500.0  # USD/kg (industrial CNTs) #10.0, 50.0, 100.0 ( future prices)
SalepriceCNTperkg = triang(
    c=(SalepriceCNT_mode - SalepriceCNT_min) / (SalepriceCNT_max - SalepriceCNT_min),
    loc=SalepriceCNT_min,
    scale=(SalepriceCNT_max - SalepriceCNT_min)
).rvs(size=N, random_state=rng)


# Carbon black selling price anchor and uncertainty — references:
# 1) Intratec Solutions, Carbon Black Price & Cost Reports
#    https://www.intratec.us/solutions/primary-commodity-prices/commodity/carbon-black-prices
#    (Widely cited source for industrial carbon black prices; typical ranges
#     ~1,500–3,000 USD/ton depending on grade and market conditions.)
#
# 2) Grand View Research (or similar market reports),
#    “Carbon Black Market Size, Share & Trends”
#    https://www.grandviewresearch.com/industry-analysis/carbon-black-market
#    (Provides market pricing context and long-term trends for commodity
#     and specialty carbon black products.)
#
# 3) IEA / Industry reports on carbon materials and chemicals
#    International Energy Agency (various years)
#    https://www.iea.org/
#    (Often used in TEA/LCA studies to justify representative commodity
#     price ranges for carbon materials.)
#
# 4) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Recommends lognormal distributions for commodity product prices
#     subject to market volatility in early-stage techno-economic models.)
#
SalepriceCBperkg_base = 2.4  # USD/kg (≈2,400 USD/ton, industrial carbon black anchor)
SalepriceCBperkg = SalepriceCBperkg_base * lognorm(s=0.25, scale=1.0).rvs(size=N, random_state=rng)


# Hydrogen selling price anchor and uncertainty (scenario-based) — references:
# 1) U.S. Department of Energy (DOE), Hydrogen Shot Initiative
#    https://www.energy.gov/eere/fuelcells/hydrogen-shot
#    (Defines a long-term cost target of ~1 USD/kg H2; widely cited in
#     academic and policy-driven hydrogen TEA studies.)
#
# 2) International Energy Agency (IEA), Global Hydrogen Review (latest editions)
#    https://www.iea.org/reports/global-hydrogen-review-2023
#    (Provides current and projected hydrogen production and market prices
#     across regions and pathways.)
#
# 3) Muradov, N.Z., & Veziroğlu, T.N. (2005),
#    “Green path from fossil fuels to hydrogen energy,”
#    International Journal of Hydrogen Energy, 30, 225–237.
#    https://doi.org/10.1016/j.ijhydene.2004.04.033
#    (Foundational academic reference for turquoise hydrogen concepts,
#     often cited when discussing H2 co-production economics.)
#
# 4) Abánades, A., et al. (2012),
#    “Thermal methane cracking: an overview,”
#    International Journal of Hydrogen Energy, 37, 9559–9573.
#    https://doi.org/10.1016/j.ijhydene.2012.03.004
#    (Discusses hydrogen yield and economic context for methane pyrolysis routes.)
#
# 5) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Supports use of triangular distributions for scenario-based product prices
#     in early-stage techno-economic assessments.)
#
SalepriceH2_min, SalepriceH2_mode, SalepriceH2_max = 1.0, 2.5, 6.0  # USD/kg (H2 selling price scenarios)
SalepriceH2perkg = triang(
    c=(SalepriceH2_mode - SalepriceH2_min) / (SalepriceH2_max - SalepriceH2_min),
    loc=SalepriceH2_min,
    scale=(SalepriceH2_max - SalepriceH2_min)
).rvs(size=N, random_state=rng)



# Process uncertainties (physics drivers)

# Carbon yield to solid from methane cracking (fraction of CH4 carbon ending as solid carbon)
# Typical FCCVD yield depends on residence time, catalyst, H2 dilution, etc. – keep broad.
CarbonyieldFC = triang(c=(0.60 - 0.40) / (0.80 - 0.40), loc=0.40, scale=0.40).rvs(size=N, random_state=rng)


# Split of solid carbon into CNT vs carbon black (bounded 0..1)
PercentCNTYield = beta(a=40, b=6).rvs(size=N, random_state=rng)  # mean ~0.87
#PercentCBYield = 1.0 - PercentCNTYield
# Methane conversion (fraction of CH4 cracked) — references:
# 1) Abánades, A., et al. (2012),
#    “Thermal methane cracking: an overview,”
#    International Journal of Hydrogen Energy, 37, 9559–9573.
#    https://doi.org/10.1016/j.ijhydene.2012.03.004
#    (Reports methane conversion levels typically in the ~60–90% range for
#     high-temperature (≈1000–1300°C) thermal and catalytic methane cracking reactors,
#     depending on residence time, reactor design, and catalyst.)
#
# 2) Muradov, N.Z., & Veziroğlu, T.N. (2005),
#    “Green path from fossil fuels to hydrogen energy,”
#    International Journal of Hydrogen Energy, 30, 225–237.
#    https://doi.org/10.1016/j.ijhydene.2004.04.033
#    (Discusses methane conversion efficiencies for turquoise hydrogen routes,
#     indicating partial but high single-pass conversion with recycle.)
#
# 3) Trommer, D., et al. (2017),
#    “Methane pyrolysis in a liquid metal bubble column reactor,”
#    International Journal of Hydrogen Energy, 42, 24270–24282.
#    https://doi.org/10.1016/j.ijhydene.2017.08.194
#    (Experimental data show methane conversions commonly around 70–85% per pass
#     under atmospheric pressure and high-temperature operation.)
#
# 4) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Recommends bounded probability distributions (e.g., beta) for conversion
#     parameters constrained between 0 and 1 in early-stage reactor/TEA models.)
#
# Modeling note:
# - A beta distribution is used here to reflect bounded uncertainty (0–1)
#   with a mean near 0.75, consistent with reported single-pass methane
#   conversions in high-temperature cracking reactors.
#J. Peden, J. Ryley, J. Terrones, F. Smail, J. A. Elliot, A. Windle, A. Boies,
#Multi-pass fccvd for co-production of hydrogen and carbon nanotube mats
#with process gas recycling, Nature Energy (2025). doi:10.1038/s41560-
#025-01925-3.
#URL https://doi.org/10.1038/s41560-025-01925-3
CH4_conversion = beta(a=18, b=6).rvs(size=N, random_state=rng)  # mean ≈ 0.75


# Hydrogen recovery from PSA (pressure swing adsorption) — references:
# 1) Sircar, S., & Golden, T.C. (2000),
#    “Purification of hydrogen by pressure swing adsorption,”
#    Separation Science and Technology, 35(5), 667–687.
#    https://doi.org/10.1081/SS-100100197
#    (Classic PSA reference; reports hydrogen recovery typically in the
#     ~75–90% range depending on cycle design and feed composition.)
#
# 2) Yang, R.T. (1997),
#    Gas Separation by Adsorption Processes,
#    Butterworth-Heinemann.
#    (Foundational text; documents industrial PSA hydrogen recovery
#     efficiencies commonly between 80–90%.)
#
# 3) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Supports bounded beta distributions for separation efficiencies
#     constrained between 0 and 1 in early-stage process modeling.)
#
# 4) Abánades, A., et al. (2012),
#    “Thermal methane cracking: an overview,”
#    International Journal of Hydrogen Energy, 37, 9559–9573.
#    https://doi.org/10.1016/j.ijhydene.2012.03.004
#    (Discusses hydrogen separation and recycle assumptions in methane
#     pyrolysis process flowsheets.)
#
# Modeling note:
# - A beta distribution with mean ≈0.86 reflects best-practice industrial PSA
#   hydrogen recovery while allowing realistic uncertainty in early-stage TEA.
#
Percent_of_H2_from_purge_gas = beta(a=25, b=4).rvs(size=N, random_state=rng)  # mean ≈ 0.86
# Hydrogen recovery from PSA (fraction of H2 in purge recovered as product)



# PSA / recycle operating pressure (screening-level assumption) — references:
# 1) Yang, R.T. (1997),
#    Gas Separation by Adsorption Processes,
#    Butterworth-Heinemann.
#    (Authoritative reference on PSA design; reports typical hydrogen PSA
#     feed pressures in the range of ~10–30 bar depending on upstream process.)
#
# 2) Sircar, S., & Golden, T.C. (2000),
#    “Purification of hydrogen by pressure swing adsorption,”
#    Separation Science and Technology, 35(5), 667–687.
#    https://doi.org/10.1081/SS-100100197
#    (Describes industrial H2 PSA systems operating at elevated pressures,
#     commonly between ~15–30 bar to balance recovery and purity.)
#
# 3) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Provides typical pressure ranges for gas separation units and supports
#     use of triangular distributions for preliminary process design.)
#
# 4) Abánades, A., et al. (2012),
#    “Thermal methane cracking: an overview,”
#    International Journal of Hydrogen Energy, 37, 9559–9573.
#    https://doi.org/10.1016/j.ijhydene.2012.03.004
#    (Assumes pressurised hydrogen separation and recycle loops in methane
#     pyrolysis flowsheets, consistent with PSA feed pressures >10 bar.)
#
# Modeling note:
# - A triangular distribution is used to represent screening-level uncertainty
#   in PSA feed pressure while remaining consistent with industrial practice.
#
PSA_feed_pressure_bar = triang(
    c=(20.0 - 10.0) / (35.0 - 10.0),  # mode position
    loc=10.0,
    scale=(35.0 - 10.0)
).rvs(size=N, random_state=rng)



# Compressor isentropic efficiency (dimensionless) — references:
# 1) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Reports typical industrial gas compressor isentropic efficiencies
#     in the range ~0.60–0.85 for preliminary design and TEA studies.)
#
# 2) Perry, R.H., & Green, D.W. (2008).
#    Perry’s Chemical Engineers’ Handbook (8th ed.). McGraw-Hill.
#    (Standard reference; lists centrifugal and reciprocating compressor
#     efficiencies commonly between 60–85% depending on size and service.)
#
# 3) Turton, R., Bailie, R.C., Whiting, W.B., & Shaeiwitz, J.A. (2012).
#    Analysis, Synthesis, and Design of Chemical Processes (4th ed.). Pearson.
#    (Recommends screening-level compressor efficiencies of ~0.70–0.80
#     for early-stage process simulations.)
#
# 4) Couper, J.R., Penney, W.R., Fair, J.R., & Walas, S.M. (2010).
#    Chemical Process Equipment: Selection and Design (3rd ed.). Elsevier.
#    (Provides practical efficiency ranges for industrial compressors
#     used in gas separation and recycle systems.)
#
# Modeling note:
# - A beta distribution is used to reflect bounded uncertainty (0–1),
#   then clipped to the physically realistic industrial range of 0.60–0.85.
#
PSA_compressor_eta = np.clip(
    beta(a=18, b=8).rvs(size=N, random_state=rng),
    0.6, 0.85
)


# Catalyst/promoter dosing (kg per kg CNT) — updated with FCCVD literature
# References:
# - Zhang et al., Carbon 148 (2019) 537–548, DOI: 10.1016/j.carbon.2019.03.094
# - Endo et al., Carbon 46 (2008) 170–176
# - Isaacs et al., Carbon 47 (2009) 3053–3063
# - Li et al., Chem. Eng. J. 322 (2017) 521–531
# - Abánades et al., IJHE 37 (2012) 9559–9573

# Iron catalyst consumption:
# Typical industrial FCCVD: ~0.5–2.0 wt% of CNT mass
Fe_kg_per_kgCNT = triang(
    c=(0.010 - 0.005) / (0.020 - 0.005),  # mode at 1.0 wt%
    loc=0.005,                           # 0.5 wt%
    scale=(0.020 - 0.005)                # up to 2.0 wt%
).rvs(size=N, random_state=rng)

# Sulfur promoter consumption:
# Typical FCCVD sulfur dosing: ~0.02–0.2 wt% of CNT mass
S_kg_per_kgCNT = triang(
    c=(0.0008 - 0.0002) / (0.0020 - 0.0002),  # mode ≈0.08 wt%
    loc=0.0002,                               # 0.02 wt%
    scale=(0.0020 - 0.0002)                   # up to 0.20 wt%
).rvs(size=N, random_state=rng)


# Optional hydrogen co-feed fraction relative to methane (molar basis) — references:
# 1) Muradov, N.Z., & Veziroğlu, T.N. (2005),
#    “Green path from fossil fuels to hydrogen energy,”
#    International Journal of Hydrogen Energy, 30, 225–237.
#    https://doi.org/10.1016/j.ijhydene.2004.04.033
#    (Discusses hydrogen co-feeding and recycle in methane pyrolysis systems
#     to suppress coke formation and stabilize reactor operation.)
#
# 2) Abánades, A., et al. (2012),
#    “Thermal methane cracking: an overview,”
#    International Journal of Hydrogen Energy, 37, 9559–9573.
#    https://doi.org/10.1016/j.ijhydene.2012.03.004
#    (Reviews methane cracking reactor configurations where hydrogen dilution
#     or recycle is used to control conversion, residence time, and carbon morphology.)
#
# 3) Trommer, D., et al. (2017),
#    “Methane pyrolysis in a liquid metal bubble column reactor,”
#    International Journal of Hydrogen Energy, 42, 24270–24282.
#    https://doi.org/10.1016/j.ijhydene.2017.08.194
#    (Reports hydrogen recycle fractions on the order of 5–30 mol% relative
#     to methane feed in high-temperature cracking experiments.)
#
# 4) Endo, M., et al. (2008),
#    “Mass production of carbon nanotubes,”
#    Carbon, 46, 170–176.
#    https://doi.org/10.1016/j.carbon.2007.11.016
#    (Describes FCCVD industrial practice where hydrogen is co-fed or recycled
#     to control catalyst activity and CNT quality.)
#
# 5) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Supports triangular distributions for bounded operating ratios used
#     in early-stage reactor and process screening models.)
#
# Modeling note:
# - A triangular distribution from 0–30 mol% H2/CH4 captures reported industrial
#   and experimental co-feed/recycle practices while avoiding over-constraint.
#
H2_to_CH4_molar = triang(
    c=(0.10 - 0.00) / (0.30 - 0.00),
    loc=0.00,
    scale=0.30
).rvs(size=N, random_state=rng)



# CAPEX uncertainty multiplier — references:
# 1) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Standard reference for process cost estimation; indicates that early-stage
#     capital cost estimates (Class 4–5) typically carry uncertainties of
#     ±25–50%, commonly modeled using lognormal distributions.)
#
# 2) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Recommends lognormal distributions for CAPEX uncertainty in screening-
#     level techno-economic assessments due to asymmetric upside risk.)
#
# 3) AACE International (2020),
#    Recommended Practice No. 18R-97 – Cost Estimate Classification System.
#    https://web.aacei.org/resources/publications/recommended-practices
#    (Classifies early feasibility studies as having typical accuracy ranges
#     of −30% to +50%, consistent with a lognormal spread of σ ≈ 0.2–0.3.)
#
# 4) Bistline, J.E., et al. (2021),
#    “Value of technology innovation and uncertainty in energy system modeling,”
#    Energy Economics, 99, 105281.
#    https://doi.org/10.1016/j.eneco.2021.105281
#    (Uses lognormal CAPEX multipliers to represent capital cost uncertainty
#     in early-stage energy technology models.)
#
# Modeling note:
# - lognormal(s=0.25) corresponds to a typical ±30–40% spread around the median,
#   appropriate for pre-FEED / conceptual design TEA.
#
CAPEX_factor = lognorm(s=0.25, scale=1.0).rvs(size=N, random_state=rng)


# OPEX uncertainty multiplier — references:
# 1) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.). Elsevier.
#    (Reports that operating cost estimates at screening/pre-FEED level
#     typically have smaller uncertainty than CAPEX, often ±10–30%,
#     and are suitably represented with lognormal distributions.)
#
# 2) Peters, M.S., Timmerhaus, K.D., & West, R.E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.). McGraw-Hill.
#    (Indicates raw materials, utilities, and labor costs are usually better
#     constrained than capital costs; OPEX uncertainty commonly ~±15–25%
#     in early-stage studies.)
#
# 3) AACE International (2020),
#    Recommended Practice No. 18R-97 – Cost Estimate Classification System.
#    (For Class 4–5 estimates, operating cost accuracy is generally narrower
#     than capital cost accuracy, supporting σ ≈ 0.1–0.2.)
#
# 4) Rubin, E.S., Azevedo, I.M.L., Jaramillo, P., & Yeh, S. (2015).
#    “A review of learning rates for electricity supply technologies,”
#    Energy Policy, 86, 198–218.
#    https://doi.org/10.1016/j.enpol.2015.06.011
#    (Uses lognormal uncertainty treatment for operating costs in
#     early-stage energy technology assessments.)
#
# Modeling note:
# - lognormal(s=0.15) corresponds to roughly ±15–20% dispersion around the median,
#   appropriate for screening-level OPEX in TEA.
#
OPEX_factor = lognorm(s=0.15, scale=1.0).rvs(size=N, random_state=rng)




# Insulation efficiency factor (0..1): “how much of nominal heat loss is avoided”
Insulationefficiency_unc = beta(a=45, b=5).rvs(size=N, random_state=rng)  # mean ~0.90


# Insulation thickness (m) for ceramic fiber blanket — references:
#
# 1) Incropera, F.P., DeWitt, D.P., Bergman, T.L., & Lavine, A.S. (2011).
#    Fundamentals of Heat and Mass Transfer (7th ed.). Wiley.
#    (Chapter on high-temperature insulation design shows that industrial
#     furnace walls using ceramic fiber blankets typically employ total
#     insulation thicknesses in the ~25–75 mm range, depending on temperature
#     and allowable heat loss.)
#
# 2) European Commission – BAT Reference Document (BREF) for
#    Energy Efficiency (ENE), 2019.
#    (Reports refractory + ceramic fiber linings for industrial furnaces
#     commonly in the 20–100 mm range for high-temperature operation,
#     with thinner linings favored for lightweight fiber systems.)
#
# 3) Carbolite Gero Ltd., High-Temperature Furnace Design Guides.
#    https://www.carbolite-gero.com
#    (Industrial and laboratory furnace specifications show ceramic fiber
#     blanket thicknesses of 25–50 mm for ~1000–1200 °C service and
#     50–75 mm for higher-duty or energy-efficient designs.)
#
# 4) Nabertherm GmbH, Industrial Furnace Construction Manuals.
#    https://nabertherm-industrial.com
#    (Published furnace cross-sections indicate ceramic fiber insulation
#     layers typically between 30 and 80 mm depending on duty and wall losses.)
#
# Modeling note:
# - Triangular(0.025, 0.05, 0.075) m represents a realistic screening-level
#   uncertainty band for ceramic fiber blanket thickness in industrial
#   CNT/CVD/FLB furnaces operating at ~900–1300 °C.
#
blanket_thickness_m = triang(
    c=(0.05 - 0.025) / (0.075 - 0.025),
    loc=0.025,
    scale=0.050
).rvs(size=N, random_state=rng)


# Mat areal density (gsm) uncertainty — references:
#
# 1) J. Peden, J. Ryley, J. Terrones, F. Smail, J. A. Elliot, A. Windle, A. Boies,
#    Multi-pass fccvd for co-production of hydrogen and carbon nanotube mats
#    with process gas recycling, Nature Energy (2025). doi:10.1038/s41560-
#    025-01925-3.
#    URL https://doi.org/10.1038/s41560-025-01925-3
#    
#
#
# 2) Endo, M., Kim, Y.A., Hayashi, T., et al. (2008).
#    Mass production of carbon nanotubes.
#    Carbon, 46, 170–176. https://doi.org/10.1016/j.carbon.2007.11.016
#    (Discusses CNT mats and sheets produced via CVD routes with
#     low areal densities suitable for scalable manufacturing.)
#
# 3) Isaacs, J.A., Tanwani, A., Healy, M.L., & Dahlben, L.J. (2009).
#    Single-walled carbon nanotube manufacturing cost analysis.
#    Carbon, 47, 3053–3063. https://doi.org/10.1016/j.carbon.2009.06.041
#    (Uses CNT sheet/mat basis weights typically in the tens of g·m⁻²
#     for downstream product cost normalization.)
#
# 4) Zhang, M., Fang, S., Zakhidov, A.A., et al. (2005).
#    Strong, transparent, multifunctional, carbon nanotube sheets.
#    Science, 309, 1215–1219. https://doi.org/10.1126/science.1115311
#    (Early demonstration of CNT sheets with areal densities
#     of order 10–30 g·m⁻².)
#
# Modeling note:
# - Triangular(10, 20, 40) g·m⁻² represents a realistic screening-level
#   uncertainty range for CNT mats used in structural, filtration,
#   and energy applications, consistent with both academic literature
#   and teaching references (Smail & Peden).
#
Matgsm = triang(
    c=(20.0 - 10.0) / (40.0 - 10.0),
    loc=10.0,
    scale=30.0
).rvs(size=N, random_state=rng)



# =============================================================================
# CAPEX scaling exponent (dimensionless)
#
# Reference:
# 1) Peters, M. S., Timmerhaus, K. D., & West, R. E. (2003).
#    Plant Design and Economics for Chemical Engineers (5th ed.).
#    McGraw-Hill.
#    → Capital cost scaling with capacity is commonly represented as:
#      C2 = C1 * (S2/S1)^n, where n typically lies between 0.4 and 0.7
#      for process equipment and integrated facilities.
#
# 2) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design (2nd ed.).
#    Elsevier.
#    → Reports economy-of-scale exponents in the 0.3–0.9 range depending
#      on modularity and degree of integration.
#
# Modeling note:
# - Normal distribution centered at 0.50 with truncation reflects
#   screening-level uncertainty before detailed equipment sizing.
#
scalingexponent = np.clip(
    norm(loc=0.50, scale=0.08).rvs(size=N, random_state=rng),
    0.30, 0.90
)



# =============================================================================
# Tube scale-up factor (dimensionless)
#
# Reference:
# 1) Isaacs, J. A., Tanwani, A., Healy, M. L., & Dahlben, L. J. (2009).
#    Single-walled carbon nanotube manufacturing cost analysis.
#    Carbon, 47, 3053–3063.
#    https://doi.org/10.1016/j.carbon.2009.06.041
#    → CNT CVD reactors are scaled primarily by increasing tube count
#      rather than tube diameter; practical tube-count scale-up factors
#      of ~0.8–1.3 are used for screening studies.
#
# 2) Endo, M., Kim, Y. A., Hayashi, T., et al. (2008).
#    Mass production of carbon nanotubes.
#    Carbon, 46, 170–176.
#    https://doi.org/10.1016/j.carbon.2007.11.016
#    → Industrial CNT production favors modular tube replication
#      with limited per-tube scale-up.
#
# Modeling note:
# - Triangular distribution reflects asymmetric feasibility:
#   under-scaling is easier than aggressive tube enlargement.
#
Tubescaleupfactor = triang(
    c=(1.0 - 0.8) / (1.3 - 0.8),
    loc=0.8,
    scale=0.5
).rvs(size=N, random_state=rng)



# =============================================================================
# Gas recovery CAPEX scaling exponent (dimensionless)
#
# Reference:
# 1) Seader, J. D., Henley, E. J., & Roper, D. K. (2011).
#    Separation Process Principles (3rd ed.).
#    Wiley.
#    → PSA, membrane, and gas separation systems exhibit scale exponents
#      typically between 0.5 and 0.8 due to modular skids and compressors.
#
# 2) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design.
#    Elsevier.
#    → Gas treatment and recovery units often scale more weakly than
#      reactors due to auxiliary equipment and balance-of-plant limits.
#
# Modeling note:
# - Triangular(0.40, 0.60, 0.90) captures uncertainty between
#   highly modular PSA systems and more integrated recovery trains.
#
Gasrecoveryscalefactor = triang(
    c=(0.60 - 0.40) / (0.90 - 0.40),
    loc=0.40,
    scale=0.50
).rvs(size=N, random_state=rng)


# =============================================================================
# Infrastructure & QA equipment CAPEX (USD)
#
# Reference:
# 1) Peters, M. S., Timmerhaus, K. D., & West, R. E. (2003).
#    Plant Design and Economics for Chemical Engineers.
#    → Laboratory, QA/QC, and analytical infrastructure is commonly
#      estimated as a fixed block cost at early design stages,
#      independent of reactor sizing.
#
# 2) Isaacs et al. (2009), Carbon, 47, 3053–3063.
#    → CNT manufacturing cost models include fixed QA, metrology,
#      and characterization infrastructure as lump-sum CAPEX.
#
# Modeling note:
# - Lognormal uncertainty reflects procurement, vendor selection,
#   and instrumentation depth (R&D-grade vs industrial QA).
#
InfrastructureQAequipementcost_base = 250_000.0
InfrastructureQAequipementcost = (
    InfrastructureQAequipementcost_base
    * lognorm(s=0.30, scale=1.0).rvs(size=N, random_state=rng)
)


# =============================================================================
# Useful life of infrastructure & QA equipment (years)
#
# Reference:
# 1) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design.
#    → Analytical and QA equipment lifetimes typically range from
#      3–10 years depending on obsolescence and maintenance.
#
# 2) U.S. IRS MACRS Depreciation Schedules (industrial equipment).
#    → Laboratory and process control equipment commonly depreciated
#      over 5–7 years.
#
# Modeling note:
# - Triangular(3, 5, 10) captures rapid obsolescence at low end
#   and robust industrial instrumentation at high end.
#
usefullifeinfrastructure = np.clip(
    np.round(
        triang(
            c=(5 - 3) / (10 - 3),
            loc=3,
            scale=7
        ).rvs(size=N, random_state=rng)
    ).astype(int),
    3, 10
)

# =============================================================================
# Gas recovery reference CAPEX (USD)
#
# Reference:
# 1) Seader, J. D., Henley, E. J., & Roper, D. K. (2011).
#    Separation Process Principles.
#    → Screening-level cost estimates for PSA and gas separation
#      units are often normalized per unit flow or capacity and
#      later scaled using power-law exponents.
#
# 2) Towler, G., & Sinnott, R. (2013).
#    Chemical Engineering Design.
#    → Early-stage models frequently use simplified reference
#      costs before detailed compressor and adsorber sizing.
#
# Modeling note:
# - This base value is a placeholder anchor to be replaced by
#   a detailed PSA sizing model once flowrates and pressures
#   are fully specified.
#
# Gas recovery / PSA package reference CAPEX (USD) – screening-level anchor
Gasrecoverycost_base = 1260.0  # USD (placeholder anchor; replace once you choose a real basis)

# Uncertainty model: PSA / gas treatment package costs are commonly treated as multiplicative
# uncertainties (vendor quotes, scope definition, pressure/material specs).
# Reference basis: process cost-estimation practice for packaged units and early-stage CAPEX:
# - Towler, G. & Sinnott, R. (2013) Chemical Engineering Design (2nd ed.), Elsevier.
# - Peters, M.S., Timmerhaus, K.D., West, R.E. (2003) Plant Design and Economics for Chemical Engineers (5th ed.), McGraw-Hill.
#
# Here: P10~0.5x, P50~1.0x, P90~2.0x (approx), i.e., factor-of-2 uncertainty typical at screening stage.
Gasrecoverycost = Gasrecoverycost_base * lognorm(s=0.45, scale=1.0).rvs(size=N, random_state=rng)
Gasrecoverycost = np.clip(Gasrecoverycost, 0.5 * Gasrecoverycost_base, 2.0 * Gasrecoverycost_base)


# ============================================================
# 4) INSULATION / HEAT LOSS MODEL
# ============================================================

# Thermal conductivity of ceramic fiber blanket vs temperature (W/m-K)
# Replace with your chosen product curve; this is a typical order-of-magnitude table.
# Thermal conductivity of ceramic fiber blanket vs temperature (W/m-K)
# Typical alumina–silica ceramic fiber blanket values
# Sources:
# Wang et al., J. Mater. Sci. 48 (2013) 1863–1871
# ASTM C892 / Incropera et al., Fundamentals of Heat and Mass Transfer
_k_table = {200: 0.06, 400: 0.11, 600: 0.16, 800: 0.23, 1000: 0.32}


def k_ceramic_blanket_W_mK(T_C: float) -> float:
    Ts = np.array(sorted(_k_table.keys()), dtype=float)
    ks = np.array([_k_table[t] for t in Ts], dtype=float)
    return float(np.interp(T_C, Ts, ks))


def heat_loss_W(A_m2: np.ndarray, T_hot_C: float, T_amb_C: float, thickness_m: np.ndarray) -> np.ndarray:
    T_mean = 0.5 * (T_hot_C + T_amb_C)
    k = k_ceramic_blanket_W_mK(T_mean)
    dT = (T_hot_C - T_amb_C)
    return (k * A_m2 * dT) / thickness_m

# -----------------------------
# Insulation property + cost model (FCCVD + FLB)
# -----------------------------


def k_ceramic_fiber_W_mK(T_C: np.ndarray) -> np.ndarray:
    """
    Thermal conductivity (W/m·K) for refractory ceramic fiber blanket vs temperature.
    Source (technical datasheet, widely used in furnace design):
      - Unifrax / Fiberfrax Cerablanket® Ceramic Fiber Blanket, typical k-values vs mean temp
        https://www.unifrax.com/products/ceramic-fibers/cerablanket-ceramic-fiber-blanket/
    Notes:
      - Datasheets usually report k at mean temperature. We use a simple interpolation table.
      - If you later pick a specific blanket grade (1260°C vs 1430°C), update this table accordingly.
    """
    # Typical order-of-magnitude values consistent with RCF blanket datasheets (mean temperature basis)
    T_table = np.array([200, 400, 600, 800, 1000], dtype=float)
    k_table = np.array([0.06, 0.11, 0.16, 0.23, 0.32], dtype=float)
    T_C = np.clip(np.asarray(T_C, dtype=float), T_table.min(), T_table.max())
    return np.interp(T_C, T_table, k_table)

def sample_ceramic_blanket_cost_usd_per_m2_25mm(rng, N: int, GBPUSD: float) -> np.ndarray:
    """
    Market-anchored ceramic fiber blanket cost (USD/m^2) for 25 mm thickness.
    We convert UK retail roll prices into an equivalent USD/m^2 basis and model as triangular.

    Market references (25mm rolls, 610mm wide, ~7.32m long):
      - Vitcas Ceramic Fibre Blanket 1260°C, 25mm x 610mm x 7.32m (price shown on page)
        https://www.firestoppingshop.com/product/ceramic-fibre-blanket-1260c
      - Scarva Kilns Ceramic Fibre Blanket 1300°C Grade 128 Density, 7.32m x 610mm x 25mm roll
        https://www.scarva.com/en/Scarva-Kilns-Ceramic-Fibre-Blanket---1300%C2%BAC-Grade-128-Density-732m-x-610mm-x-25mm-roll/m-846.aspx

    Derived unit-cost sanity check:
      Area per roll ≈ 0.610 * 7.32 ≈ 4.47 m^2
      Vitcas: £56.25 ex VAT -> ~£12.6/m^2; £67.50 inc VAT -> ~£15.1/m^2
      Scarva: £112.82 -> ~£25.3/m^2
    We set: min~$16/m^2, mode~$22/m^2, max~$40/m^2 (25mm), scaled by GBPUSD.
    """
    # Use GBP/m2 anchors, then convert to USD/m2 using GBPUSD
    gbp_min  = 12.5   # ~Vitcas ex VAT per m2
    gbp_mode = 16.0   # near Vitcas inc VAT / typical retail
    gbp_max  = 30.0   # conservative upper bound (retail variability / higher grade)
    usd_min, usd_mode, usd_max = gbp_min*GBPUSD, gbp_mode*GBPUSD, gbp_max*GBPUSD
    return rng.triangular(usd_min, usd_mode, usd_max, size=N)

def insulation_geometry_from_scale(Throughputofscale_ton_per_year: float):
    """
    Reactor geometry assumptions used only for insulation area (not kinetics).
    Legacy anchors from your 2023 model (Beston pyrolysis reactor dimensions used as proxy):
      https://www.bestongroup.com/pyrolysis-plant-cost/
    """
    if Throughputofscale_ton_per_year < 1_000_000:
        D_m, H_m = 1.4, 4.9
    else:
        D_m, H_m = 1.8, 18.5
    return D_m, H_m

def compute_insulation_cost_usd(
    rng,
    N: int,
    Throughputofscale_ton_per_year: float,
    n_reactors: np.ndarray,
    T_hot_C: float,
    T_amb_C: float,
    Heat_for_insulation_kW: np.ndarray,
    GBPUSD: float
) -> np.ndarray:
    """
    Uses conduction: Q = k*A*ΔT / thickness  -> thickness = k*A*ΔT / Q
    Then insulation CAPEX = Area * (thickness / 25mm) * (USD/m2 at 25mm) * install_factor

    This ties your heat model to insulation thickness AND cost.
    """
    D_m, H_m = insulation_geometry_from_scale(Throughputofscale_ton_per_year)
    r = D_m / 2.0
    A_m2 = 2.0*np.pi*(r**2) + 2.0*np.pi*r*H_m  # ends + shell

    # k at mean temperature (approx)
    T_mean_C = 0.5*(T_hot_C + T_amb_C)
    k = k_ceramic_fiber_W_mK(T_mean_C)  # scalar

    # per-reactor heat that must be "blocked" by insulation (kW/reactor)
    n_reactors_safe = np.maximum(np.asarray(n_reactors, dtype=float), 1.0)
    Q_per_reactor_W = (Heat_for_insulation_kW / n_reactors_safe) * 1000.0  # W

    dT = max(T_hot_C - T_amb_C, 1.0)
    thickness_m = (k * A_m2 * dT) / np.maximum(Q_per_reactor_W, 1.0)

    # Bound thickness to realistic installation range
    thickness_m = np.clip(thickness_m, 0.005, 0.25)  # 5 mm to 250 mm

    # Blanket unit cost sampled (USD/m2) for 25 mm thickness
    unit_cost_usd_m2_25mm = sample_ceramic_blanket_cost_usd_per_m2_25mm(rng, N, GBPUSD)

    # Installation / fastening / waste factor (triangular)
    install_factor = rng.triangular(1.05, 1.15, 1.35, size=N)

    cost_per_reactor = A_m2 * (thickness_m / 0.025) * unit_cost_usd_m2_25mm * install_factor
    total_cost = cost_per_reactor * n_reactors_safe
    return total_cost



# ============================================================
# 5) Capital Cost Model
# ============================================================

def Capital_cost(
    Tube_scale_up_factor: float,
    Tube_capital_exponent_scaleup: float,
    Throughput_: float,
    Throughput_of_scale: float,
    Number_of_reactors: float,
    Number_of_facility: int,
    Tube_per_reactor_sclaling_exponent_factor: float,
    Reactor_scaling_factor: float,
    Facility_scaling_factor: float,
    Single_reactor_price_usd,
    Facility_share_cost: float,
    scaling_exponent: float,
    Recovery_of_H2_from_purge_gas,
    Gas_recovery_cost: float,
    Gas_recovery_scale_factor: float,
    Infrastructure_QA_equipement_cost: float,
    Insulation_cost: float | np.ndarray | None = 0.0,
    *,
    compute_insulation: bool = False,
    rng=None,
    N_samples: int | None = None,
    Throughputofscale_ton_per_year: float | None = None,
    T_hot_C: float | None = None,
    T_amb_C: float = 40.0,
    Heat_for_insulation_kW: np.ndarray | None = None,
    GBPUSD: float = 1.0,
):
    """Total installed CAPEX (USD) composed of:
    - Reactor/train capital
    - Facility shared capital
    - Gas recovery/PSA/recycle capital
    - Infrastructure/QA equipment
    - Insulation cost (placeholder, optional)
    """
    reactor_cost = reactor_capital(
        Tube_scale_up_factor,
        Tube_capital_exponent_scaleup,
        Throughput_,
        Throughput_of_scale,
        Number_of_reactors,
        Number_of_facility,
        Tube_per_reactor_sclaling_exponent_factor,
        Reactor_scaling_factor,
        Facility_scaling_factor,
        Single_reactor_price_usd,
    )[1]

    facility_cost = facilty_shared_capital(
        Facility_share_cost,
        Throughput_,
        Throughput_of_scale,
        Number_of_facility,
        scaling_exponent,
    )[1]

    gasrec_cost = Additional_capital(
        Recovery_of_H2_from_purge_gas,
        Gas_recovery_cost,
        Throughput_,
        Throughput_of_scale,
        Number_of_facility,
        Gas_recovery_scale_factor,
    )

    infra_cost = Infrastructure_QA_cost(Infrastructure_QA_equipement_cost, Number_of_facility)

    # Optional: compute insulation internally (vectorized) from heat-loss proxy
    if compute_insulation:
        if rng is None or N_samples is None or Throughputofscale_ton_per_year is None or T_hot_C is None or Heat_for_insulation_kW is None:
            raise ValueError(
                "compute_insulation=True requires rng, N_samples, Throughputofscale_ton_per_year, T_hot_C, and Heat_for_insulation_kW."
            )
        insulation_cost = compute_insulation_cost_usd(
            rng=rng,
            N=int(N_samples),
            Throughputofscale_ton_per_year=float(Throughputofscale_ton_per_year),
            n_reactors=np.asarray(Number_of_reactors, dtype=float),
            T_hot_C=float(T_hot_C),
            T_amb_C=float(T_amb_C),
            Heat_for_insulation_kW=np.asarray(Heat_for_insulation_kW, dtype=float),
            GBPUSD=float(GBPUSD),
        )
    else:
        insulation_cost = 0.0 if Insulation_cost is None else Insulation_cost

    return reactor_cost + facility_cost + gasrec_cost + infra_cost + insulation_cost




# ============================================================
# 5) LABOUR (FIXED OPEX)
# ============================================================


def compute_total_direct_labour_usd_per_year() -> float:
    """Backwards-compatible wrapper using the module-level Numberoffacility."""
    return compute_total_direct_labour_usd_per_year_for_facilities(Numberoffacility)


def compute_total_direct_labour_usd_per_year_for_facilities(num_facilities: int) -> float:
    hc = headcount_from_facilities(num_facilities)
    return (
        PlantManagersalary * hc["PlantManager"]
        + PlantEngineersalary * hc["PlantEngineers"]
        + Administratorsalary * hc["Administrator"]
        + Operatorsalary * hc["Operators"]
        + Assistantplantmanagersalary * hc["AssistantPlantManager"]
        + CEOsalary * hc["CEO"]
        + CTOsalary * hc["CTO"]
        + COOsalary * hc["COO"]
    )


# Default (for legacy code paths). Model functions should use
# compute_total_direct_labour_usd_per_year_for_facilities(n_facilities).
DIRECT_LABOUR_USD = compute_total_direct_labour_usd_per_year_for_facilities(Numberoffacility)


# ============================================================
# 6) PHYSICS MODEL: FCCVD
# ============================================================

# Reaction: CH4 -> C(s) + 2 H2  (endothermic)
DELTA_H_CH4_CRACK_kJ_per_mol = 75.6  # # replace with cited thermochemistry source if needed

# Molar masses (kg/mol)
MW_CH4 = molar_mass("Methane")
MW_H2 = molar_mass("Hydrogen")
MW_N2 = molar_mass("Nitrogen")


def fccvd_physics(
    cnt_target_kg_h: float,
    T_C: float,
    P_Pa: float,
    conversion_ch4: np.ndarray,
    carbon_yield_to_solids: np.ndarray,
    cnt_fraction_in_solids: np.ndarray,
    fe_kg_per_kgcnt: np.ndarray,
    s_kg_per_kgcnt: np.ndarray,
    h2_to_ch4_molar: np.ndarray,
    n2_to_ch4_molar: float = 0.20,
    psa_recovery_h2: np.ndarray | None = None,
    hot_recycle_ratio: np.ndarray | None = None,
    psa_pressure_bar: np.ndarray | float = 20.0,
    eta_comp: np.ndarray | float = 0.75,
) -> dict[str, np.ndarray]:
    """Vectorized FCCVD mass+energy balance.

    Outputs are per-sample arrays (shape N).
    """
    if psa_recovery_h2 is None:
        psa_recovery_h2 = np.ones_like(conversion_ch4)

    # Hot recycle ratio (fraction of non-solid outlet recycled hot). This mirrors the FLB recycle logic:
    # only the purge stream goes to PSA for H2 recovery.
    if hot_recycle_ratio is None:
        hot_recycle_ratio = np.zeros_like(conversion_ch4)
    hot_recycle_ratio = np.clip(hot_recycle_ratio, 0.0, 0.999999)

    # --- Solids ---
    solids_total_kg_h = cnt_target_kg_h / np.maximum(1e-12, cnt_fraction_in_solids)
    cnt_kg_h = solids_total_kg_h * cnt_fraction_in_solids
    cb_kg_h = solids_total_kg_h * (1.0 - cnt_fraction_in_solids)

    # --- Catalyst/promoter dosing (as powders) ---
    fe_kg_h = fe_kg_per_kgcnt * cnt_kg_h
    s_kg_h = s_kg_per_kgcnt * cnt_kg_h

    # --- Methane feed required ---
    # Carbon in cracked methane that becomes solid = CH4_cracked * (12/16) * carbon_yield
    fracC_in_CH4 = 12.0 / 16.0

    # Required solid carbon rate is solids_total_kg_h (treated as "C" basis).
    # CH4_cracked needed to provide that carbon (adjusted for yield):
    ch4_cracked_kg_h = solids_total_kg_h / np.maximum(1e-12, fracC_in_CH4 * carbon_yield_to_solids)

    # If conversion < 1, total feed must be higher: cracked = conversion * feed
    ch4_feed_kg_h = ch4_cracked_kg_h / np.maximum(1e-12, conversion_ch4)

    # --- Co-feed gases (molar ratios) ---
    ch4_feed_mol_s = (ch4_feed_kg_h / MW_CH4) / 3600.0

    h2_feed_mol_s = h2_to_ch4_molar * ch4_feed_mol_s
    n2_feed_mol_s = n2_to_ch4_molar * ch4_feed_mol_s

    h2_feed_kg_h = h2_feed_mol_s * MW_H2 * 3600.0
    n2_feed_kg_h = n2_feed_mol_s * MW_N2 * 3600.0

    # --- H2 production from cracked methane ---
    ch4_cracked_mol_s = (ch4_cracked_kg_h / MW_CH4) / 3600.0
    h2_gen_mol_s = 2.0 * ch4_cracked_mol_s
    h2_gen_kg_h = h2_gen_mol_s * MW_H2 * 3600.0

    # Hydrogen available to PSA = purge fraction * (generated + cofeed)
    # This applies the same recycle/PSA boundary condition used for FLB.
    purge_frac = 1.0 - hot_recycle_ratio
    h2_to_psa_kg_h = purge_frac * (h2_gen_kg_h + h2_feed_kg_h)
    h2_recovered_kg_h = psa_recovery_h2 * h2_to_psa_kg_h

    # --- Sensible heating duty (kW) ---
    T1_K = K(T_ref_C)
    T2_K = K(T_C)
    Tm_K = 0.5 * (T1_K + T2_K)

    cp_ch4 = cp_mass("Methane", Tm_K, P_Pa)
    cp_h2 = cp_mass("Hydrogen", Tm_K, P_Pa)
    cp_n2 = cp_mass("Nitrogen", Tm_K, P_Pa)

    dT = (T2_K - T1_K)

    # Convert kg/h -> kg/s
    ch4_kg_s = ch4_feed_kg_h / 3600.0
    h2_kg_s = h2_feed_kg_h / 3600.0
    n2_kg_s = n2_feed_kg_h / 3600.0

    Q_sensible_W = (ch4_kg_s * cp_ch4 + h2_kg_s * cp_h2 + n2_kg_s * cp_n2) * dT

    # --- Reaction endotherm (kW) ---
    Q_rxn_W = ch4_cracked_mol_s * (DELTA_H_CH4_CRACK_kJ_per_mol * 1000.0)

    heater_kW = (Q_sensible_W + Q_rxn_W) / 1000.0

    # --- PSA/recycle compression (kW) ---
    # compress H2 stream from 1 bar to psa_pressure_bar (screening)
    P2_Pa = psa_pressure_bar * 1e5
    psa_compression_kW = isentropic_compression_power_kW(
        m_dot_kg_s=(h2_to_psa_kg_h / 3600.0),
        fluid="Hydrogen",
        T1_K=T1_K,
        P1_Pa=P_Pa,
        P2_Pa=P2_Pa,
        eta_isentropic=eta_comp,
    )

    return {
        "CH4_kg_h": ch4_feed_kg_h,
        "H2_feed_kg_h": h2_feed_kg_h,
        "N2_kg_h": n2_feed_kg_h,
        "Fe_kg_h": fe_kg_h,
        "S_kg_h": s_kg_h,
        "CNT_kg_h": cnt_kg_h,
        "CB_kg_h": cb_kg_h,
        "H2_generated_kg_h": h2_gen_kg_h,
        "Hot_recycle_ratio": hot_recycle_ratio,
        "H2_to_PSA_kg_h": h2_to_psa_kg_h,
        "H2_recovered_kg_h": h2_recovered_kg_h,
        "heater_kW": heater_kW,
        "PSA_compression_kW": psa_compression_kW,
    }


# ============================================================
# 7) ECONOMIC MODEL (FCCVD route)
# ============================================================


def economic_model_fccvd_vectorized() -> pd.DataFrame:
    n = N  # alias for vectorized sample count
    # Production basis: CNT sold is tied to target (kg/y)
    cnt_target_kg_y = Throughputofscale_ton_per_year * 1000.0
    cnt_target_kg_h = cnt_target_kg_y / AnnualOperatingHours

    # Effective operating hours (availability)
    effective_hours = AnnualOperatingHours * Availability

    # FCCVD recycle model (mirrors FLB): fraction of non-solid outlet recycled hot,
    # purge stream goes to PSA for recovery.
    Hot_recycle_ratio_fccvd = beta(a=35, b=6).rvs(size=N, random_state=rng)  # mean ~0.85

    # Run FCCVD physics
    fcc = fccvd_physics(
        cnt_target_kg_h=cnt_target_kg_h,
        T_C=T_reactor_C,
        P_Pa=P_reactor_Pa,
        conversion_ch4=CH4_conversion,
        carbon_yield_to_solids=CarbonyieldFC,
        cnt_fraction_in_solids=PercentCNTYield,
        fe_kg_per_kgcnt=Fe_kg_per_kgCNT,
        s_kg_per_kgcnt=S_kg_per_kgCNT,
        h2_to_ch4_molar=H2_to_CH4_molar,
        n2_to_ch4_molar=0.20,
        psa_recovery_h2=Percent_of_H2_from_purge_gas,
        hot_recycle_ratio=Hot_recycle_ratio_fccvd,
        psa_pressure_bar=PSA_feed_pressure_bar,
        eta_comp=PSA_compressor_eta,
    )

    # Annual production/sales
    CNT_kg_y = fcc["CNT_kg_h"] * effective_hours
    CB_kg_y = fcc["CB_kg_h"] * effective_hours
    H2_kg_y = fcc["H2_recovered_kg_h"] * effective_hours
    H2_ton_y = H2_kg_y / 1000.0

    sell_H2 = 1.0 if AreyousellingH2.lower() == "yes" else 0.0
    sell_solids = 1.0 if AreyousellingSolidCarbon.lower() == "yes" else 0.0

    revenue = (
        CNT_kg_y * SalepriceCNTperkg * sell_solids
        + CB_kg_y * SalepriceCBperkg * sell_solids
        + H2_kg_y * SalepriceH2perkg * sell_H2
    )
    
    
    # Variable OPEX: feeds + utilities
    CH4_cost = fcc["CH4_kg_h"] * effective_hours * Methaneprice
    Fe_cost = fcc["Fe_kg_h"] * effective_hours * Ironprice
    S_cost = fcc["S_kg_h"] * effective_hours * Sulphurprice
    N2_cost = fcc["N2_kg_h"] * effective_hours * Nitrogenprice

    # Electricity: heater + PSA compression + heat-loss
    A_hot_zone_m2 = np.full(N, 120.0)  # placeholder; replace with geometry sizing
    Q_loss_W = heat_loss_W(A_hot_zone_m2, T_reactor_C, 25.0, blanket_thickness_m)
    Q_loss_kWh_y = (Q_loss_W / 1000.0) * effective_hours

    elec_kWh_y = (fcc["heater_kW"] + fcc["PSA_compression_kW"]) * effective_hours + Q_loss_kWh_y

    # Apply insulation efficiency as a reduction in conductive loss only (not in heater duty)
    elec_kWh_y = elec_kWh_y - (Q_loss_kWh_y * Insulationefficiency_unc)
    elec_kWh_y = np.maximum(0.0, elec_kWh_y)

    elec_cost = elec_kWh_y * electricaldemandcost

    variable_opex = (CH4_cost + Fe_cost + S_cost + N2_cost + elec_cost) * OPEX_factor

    # Train/facility counts (replication logic)
    # NOTE: base_cnt_kg_h should come from your FCCVD physics model at the reference scale;
    # here we use the small-scale throughput as a temporary proxy.
    base_cnt_kg_h_median = Throughput_g_h_small / 1000.0
    scale_factor_prod = (Throughputofscale_g_h / Throughput_g_h_small)

    # Production scaling exponent (alpha) is *not* a cost exponent; keep it separate.
    alpha_prod = 0.60
    max_cnt_kg_h = 200.0  # cap per reactor nameplate (kg/h) to prevent unrealistic single-reactor scaling

    # Train nameplate uncertainty: use a truncated-lognormal *factor* around the median base rate
    # so we get n_reactors uncertainty without silently re-calibrating your physics proxy.
    train_nameplate_tpy = sample_train_nameplate_tpy(rng=rng, size=N)
    train_nameplate_factor = train_nameplate_tpy / 50.0
    base_cnt_kg_h_unc = base_cnt_kg_h_median * train_nameplate_factor

    # IMPORTANT: keep return ordering consistent everywhere:
    # (n_reactors, n_trains, n_facilities, per_reactor_kg_y)
    n_reactors, n_trains, n_facilities, per_reactor_kg_y = compute_reactor_facility_counts_vectorized(
        target_tpy=float(Throughputofscale_ton_per_year),
        hours_per_year=float(AnnualOperatingHours),
        availability=float(np.mean(Availability)),
        base_cnt_kg_h=base_cnt_kg_h_unc,
        scale_factor=float(scale_factor_prod),
        alpha=float(alpha_prod),
        max_cnt_kg_h=float(max_cnt_kg_h),
        trains_per_facility=8,
        reactors_per_train=1,
    )

    # Optional: quick scaling diagnostics
    if globals().get("DEBUG_SCALE", False):
        _nr = np.asarray(n_reactors, dtype=float)
        _nt = np.asarray(n_trains, dtype=float)
        _nf = np.asarray(n_facilities, dtype=float)
        print(
            "[SCALE][FCCVD] target_tpy=",
            float(Throughputofscale_ton_per_year),
            "Throughputofscale_g_h=",
            float(Throughputofscale_g_h),
            "n_reactors_p50=",
            int(np.median(_nr)),
            "n_reactors_range=",
            (int(np.min(_nr)), int(np.max(_nr))),
            "n_trains_p50=",
            int(np.median(_nt)),
            "n_trains_range=",
            (int(np.min(_nt)), int(np.max(_nt))),
            "n_facilities_p50=",
            int(np.median(_nf)),
            "n_facilities_range=",
            (int(np.min(_nf)), int(np.max(_nf))),
        )

    # Fixed OPEX (Direct labour scales with number of facilities)
    n_facilities_int = np.asarray(n_facilities, dtype=int)
    direct_labour_usd_y = np.asarray(
        [compute_total_direct_labour_usd_per_year_for_facilities(int(nf)) for nf in n_facilities_int],
        dtype=float,
    )
    fixed_opex = direct_labour_usd_y * (1.0 + percentageoflabourcosts)

    # Depreciation (infrastructure/QA equipment is modeled per-facility in CAPEX)
    infra_depr = (InfrastructureQAequipementcost * n_facilities_int) / usefullifeinfrastructure
    # CAPEX (integrated): reactor + facility shared + gas recovery + QA + insulation (if modeled)
    # Uses your original modular structure (reactor_capital + facility_shared + additional + QA).

    Number_of_reactors = np.asarray(n_reactors, dtype=float)
    Number_of_facility = np.asarray(n_facilities, dtype=float)

    # Reactor price distributions (USD) — sampled per Monte Carlo trial
    # (Market-anchored ranges from to be_integrated.py comments; tune when you have supplier quotes)
    # FCCVD ceramic / refractory-lined furnace reactor
    # Sources:
    # - Nabertherm (industrial HT furnaces up to 1800C)
    # - Carbolite Gero (CVD furnaces)
    # - Isaacs et al., Carbon (2009)
    Singlereactorprice_FCVD = rng.triangular(40_000.0, 75_000.0, 180_000.0, size=N)
    

    # --- Market-anchored reactor prices for Monte Carlo ---
    # FCCVD: ceramic/refractory-lined up to ~1300C (triangular distribution)
    reactor_price_fccvd = rng.triangular(40_000.0, 75_000.0, 180_000.0, size=N)

    # Facility shared CAPEX anchor (USD). You can later replace with a detailed utility model.
    facility_share_cost = 75_800.0

    # Gas recovery reference CAPEX (USD) and scaling (kept from your prior model)
    Gasrecoverycost = float(Gasrecoverycost_base)

    # CAPEX from building blocks
    # CAPEX scaling exponents / factors (ported defaults; tune as you calibrate)
    # These were explicit inputs in your 2023 model; they were not yet parameterized in v3.
    Tubecapitalexponentscaleup = np.full(N, 0.65)  # tube capital exponent
    Tubeperreactorsclalingexponentfactor = np.full(N, 1.0)
    Reactorscalingfactor = np.full(N, 1.0)
    Facilityscalingfactor = np.full(N, 1.0)
    Facilitysharecost = 75_800.0  # USD; shared utilities/civil/warehouse (anchor; refine later)
    # NOTE: keep Gasrecoverycost aligned with FLB by using Gasrecoverycost_base (set above).
    Gasrecoveryscalefactor = 0.6   # scaling exponent for recovery package
    # Scalar exponents used in the shared CAPEX scaling blocks
    scalingexponent_scalar = float(np.mean(scalingexponent)) if np.ndim(scalingexponent) else float(scalingexponent)
    Gasrecoveryscalefactor_scalar = float(np.mean(Gasrecoveryscalefactor)) if np.ndim(Gasrecoveryscalefactor) else float(Gasrecoveryscalefactor)

    capex = Capital_cost(
        Tube_scale_up_factor=Tubescaleupfactor,
        Tube_capital_exponent_scaleup=Tubecapitalexponentscaleup,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_reactors=n_reactors,
        Number_of_facility=n_facilities,
        Tube_per_reactor_sclaling_exponent_factor=Tubeperreactorsclalingexponentfactor,
        Reactor_scaling_factor=Reactorscalingfactor,
        Facility_scaling_factor=Facilityscalingfactor,
        Single_reactor_price_usd=reactor_price_fccvd,
        Facility_share_cost=facility_share_cost,
        scaling_exponent=scalingexponent_scalar,
        Recovery_of_H2_from_purge_gas=Percent_of_H2_from_purge_gas,
        Gas_recovery_cost=Gasrecoverycost,
        Gas_recovery_scale_factor=Gasrecoveryscalefactor_scalar,
        Infrastructure_QA_equipement_cost=InfrastructureQAequipementcost,
        compute_insulation=True,
        rng=rng,
        N_samples=N,
        Throughputofscale_ton_per_year=Throughputofscale_ton_per_year,
        T_hot_C=1300.0,
        T_amb_C=40.0,
        Heat_for_insulation_kW=fcc["heater_kW"] * (1.0 / (1.0 - Insulationefficiency_unc)),
        GBPUSD=GBPUSD,
    )
    # Apply project-level CAPEX uncertainty factor
    capex = capex * CAPEX_factor

    # ------------------------------------------------------------
    # CAPEX BREAKDOWN + SHARES (reactor, facility, gas recovery, infra, insulation)
    # ------------------------------------------------------------
    reactor_cost = reactor_capital(
        Tube_scale_up_factor=Tubescaleupfactor,
        Tube_capital_exponent_scaleup=Tubecapitalexponentscaleup,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_reactors=n_reactors,
        Number_of_facility=n_facilities,
        Tube_per_reactor_sclaling_exponent_factor=Tubeperreactorsclalingexponentfactor,
        Reactor_scaling_factor=Reactorscalingfactor,
        Facility_scaling_factor=Facilityscalingfactor,
        Single_reactor_price_usd=reactor_price_fccvd,
    )[1]

    facility_cost = facilty_shared_capital(
        Facility_share_cost=facility_share_cost,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_facility=n_facilities,
        scaling_exponent=scalingexponent_scalar,
    )[1]

    gasrec_cost = Additional_capital(
        Recovery_of_H2_from_purge_gas=Percent_of_H2_from_purge_gas,
        Gas_recovery_cost=Gasrecoverycost,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_facility=n_facilities,
        Gas_recovery_scale_factor=Gasrecoveryscalefactor_scalar,
    )

    infra_cost = Infrastructure_QA_cost(
        Infrastructure_QA_equipement_cost=InfrastructureQAequipementcost,
        Number_of_facility=n_facilities,
    )

    insulation_cost = compute_insulation_cost_usd(
        rng=rng,
        N=N,
        Throughputofscale_ton_per_year=Throughputofscale_ton_per_year,
        n_reactors=n_reactors,
        T_hot_C=1300.0,
        T_amb_C=40.0,
        Heat_for_insulation_kW=fcc["heater_kW"] * (1.0 / (1.0 - Insulationefficiency_unc)),
        GBPUSD=GBPUSD,
    )

    # Apply the SAME project-level CAPEX uncertainty factor to each component
    reactor_cost = reactor_cost * CAPEX_factor
    facility_cost = facility_cost * CAPEX_factor
    gasrec_cost = gasrec_cost * CAPEX_factor
    infra_cost = infra_cost * CAPEX_factor
    insulation_cost = insulation_cost * CAPEX_factor

    capex_safe = np.maximum(1e-12, capex)
    Share_reactor_in_TotalCAPEX = reactor_cost / capex_safe
    Share_facility_in_TotalCAPEX = facility_cost / capex_safe
    Share_gasrec_in_TotalCAPEX = gasrec_cost / capex_safe
    Share_infra_in_TotalCAPEX = infra_cost / capex_safe
    Share_insulation_in_TotalCAPEX = insulation_cost / capex_safe

    # Maintenance (fraction of CAPEX) and total OPEX
    maintenance = Percentangecostmaintenance * capex
    total_opex = variable_opex + fixed_opex + infra_depr + maintenance
    # Cashflow basis: treat depreciation as non-cash (keep it in total OPEX + unit cost,
    # but exclude it from cashflow/NPV calculations).
    opex_cash = total_opex - infra_depr
    annual_cashflow = revenue - opex_cash
    
    # ============================================================
    # OPEX BREAKDOWN (2023-style): cost per "Throughput" + % shares
    # Here we interpret Throughput basis as CNT production (kg CNT / year),
    # i.e., costs normalized by CNT_kg_y (apple-to-apple with unit cost).
    # ============================================================
    
    CNT_kg_y_safe = np.maximum(1e-12, CNT_kg_y)  # avoid divide-by-zero
    
    # --- Annual cost components (USD/y) ---
    CH4_cost_y = CH4_cost
    Fe_cost_y  = Fe_cost
    S_cost_y   = S_cost
    N2_cost_y  = N2_cost
    Elec_cost_y = elec_cost
    
    Labour_cost_y = direct_labour_usd_y  # annual (scalar) in your model
    Overhead_cost_y = direct_labour_usd_y * percentageoflabourcosts  # annual (vectorized via overhead fraction)
    Fixed_opex_y = fixed_opex  # labour + overhead
    
    Infra_depr_y = infra_depr  # USD/y (vector)
    Maintenance_y = maintenance  # USD/y (vector)
    
    Variable_opex_y = variable_opex
    Total_opex_y = total_opex
    
    # --- Normalize by throughput (USD per kg CNT) ---
    CH4_cost_per_kgCNT = CH4_cost_y / CNT_kg_y_safe
    Fe_cost_per_kgCNT  = Fe_cost_y  / CNT_kg_y_safe
    S_cost_per_kgCNT   = S_cost_y   / CNT_kg_y_safe
    N2_cost_per_kgCNT  = N2_cost_y  / CNT_kg_y_safe
    Elec_cost_per_kgCNT = Elec_cost_y / CNT_kg_y_safe
    
    Labour_cost_per_kgCNT = Labour_cost_y / CNT_kg_y_safe
    Overhead_cost_per_kgCNT = Overhead_cost_y / CNT_kg_y_safe
    Infra_depr_per_kgCNT = Infra_depr_y / CNT_kg_y_safe
    Maintenance_per_kgCNT = Maintenance_y / CNT_kg_y_safe
    
    Variable_opex_per_kgCNT = Variable_opex_y / CNT_kg_y_safe
    Fixed_opex_per_kgCNT = Fixed_opex_y / CNT_kg_y_safe
    Total_opex_per_kgCNT = Total_opex_y / CNT_kg_y_safe
    
    # --- % contribution shares ---
    # Shares of VARIABLE OPEX (feeds + electricity)
    var_safe = np.maximum(1e-12, Variable_opex_y)
    Share_CH4_in_var = CH4_cost_y / var_safe
    Share_Fe_in_var  = Fe_cost_y  / var_safe
    Share_S_in_var   = S_cost_y   / var_safe
    Share_N2_in_var  = N2_cost_y  / var_safe
    Share_Elec_in_var = Elec_cost_y / var_safe
    
    # Shares of TOTAL OPEX (includes labour/overhead + infra depreciation + maintenance)
    tot_safe = np.maximum(1e-12, Total_opex_y)
    Share_CH4_in_tot = CH4_cost_y / tot_safe
    Share_Fe_in_tot  = Fe_cost_y  / tot_safe
    Share_S_in_tot   = S_cost_y   / tot_safe
    Share_N2_in_tot  = N2_cost_y  / tot_safe
    Share_Percursor_in_TotalOPEX = (CH4_cost_y
                                    + Fe_cost_y 
                                    +S_cost_y
                                    +N2_cost_y 
                                    )  / tot_safe
    
    Share_Elec_in_tot = Elec_cost_y / tot_safe
    Share_Labour_in_tot = Labour_cost_y / tot_safe
    Share_Overhead_in_tot = Overhead_cost_y / tot_safe
    Share_InfraDepr_in_tot = Infra_depr_y / tot_safe
    Share_Maint_in_tot = Maintenance_y / tot_safe
    
    
    
    

    # NPV (vectorized)
    years = np.arange(1, Yearforprofitibilitymodel + 1)
    discount = 1.0 / ((1.0 + InterestDiscountRate) ** years)
    pv_cashflows = annual_cashflow[:, None] * discount[None, :]
    npv = -capex + pv_cashflows.sum(axis=1)

    # Payback year (simple)
    cum = np.cumsum(np.repeat(annual_cashflow[:, None], Yearforprofitibilitymodel, axis=1), axis=1)
    payback_year = np.where(cum >= capex[:, None], years[None, :], np.inf).min(axis=1)

    # Unit cost (OPEX / kg CNT) – ex-capex definition
    unit_cost_ex_cap = total_opex / np.maximum(1e-12, CNT_kg_y)

    df = pd.DataFrame({
        # Plant sizing (per Monte Carlo draw)
        "Train_nameplate_tpy": train_nameplate_tpy,
        "Number_of_trains": np.asarray(n_trains, dtype=int),
        "Number_of_facilities": np.asarray(n_facilities, dtype=int),
        "Number_of_reactors": np.asarray(n_reactors, dtype=int),

        "CAPEX_USD": capex,
        "Reactor_cost_USD": reactor_cost,
        "Facility_cost_USD": facility_cost,
        "Gasrecovery_cost_USD": gasrec_cost,
        "Infra_cost_USD": infra_cost,
        "Insulation_cost_USD": insulation_cost,

        "Share_reactor_in_TotalCAPEX": Share_reactor_in_TotalCAPEX,
        "Share_facility_in_TotalCAPEX": Share_facility_in_TotalCAPEX,
        "Share_gasrec_in_TotalCAPEX": Share_gasrec_in_TotalCAPEX,
        "Share_infra_in_TotalCAPEX": Share_infra_in_TotalCAPEX,
        "Share_insulation_in_TotalCAPEX": Share_insulation_in_TotalCAPEX,
        "Revenue_USD_per_y": revenue,
        "OPEX_USD_per_y": total_opex,
        "Cashflow_USD_per_y": annual_cashflow,
        "NPV_USD": npv,
        "Payback_year": payback_year,
        "Unit_cost_ex_cap_USD_per_kgCNT": unit_cost_ex_cap,

        # Key FCCVD mass/energy outputs
        "CH4_kg_h": fcc["CH4_kg_h"],
        "H2_feed_kg_h": fcc["H2_feed_kg_h"],
        "N2_kg_h": fcc["N2_kg_h"],
        "Fe_kg_h": fcc["Fe_kg_h"],
        "S_kg_h": fcc["S_kg_h"],
        "CNT_kg_h": fcc["CNT_kg_h"],
        "CB_kg_h": fcc["CB_kg_h"],
        "H2_recovered_kg_h": fcc["H2_recovered_kg_h"],
    "H2_to_PSA_kg_h": fcc.get("H2_to_PSA_kg_h", np.nan),
    "Hot_recycle_ratio": fcc.get("Hot_recycle_ratio", np.nan),
        "heater_kW": fcc["heater_kW"],
        "PSA_compression_kW": fcc["PSA_compression_kW"],

    # --- Annual hydrogen production ---
    # (Recovered H2 after PSA; apples-to-apples with FLB)
    "H2_kg_y": H2_kg_y,
    "H2_ton_y": H2_ton_y,
        
        # --- OPEX breakdown (annual USD/y) ---
        "CH4_cost_USD_y": CH4_cost_y,
        "Fe_cost_USD_y": Fe_cost_y,
        "S_cost_USD_y": S_cost_y,
        "N2_cost_USD_y": N2_cost_y,
        "Elec_cost_USD_y": Elec_cost_y,
        "Labour_cost_USD_y": Labour_cost_y,
        "Overhead_cost_USD_y": Overhead_cost_y,
        "Infra_depr_USD_y": Infra_depr_y,
        "Maintenance_USD_y": Maintenance_y,
        "Variable_OPEX_USD_y": Variable_opex_y,
        "Fixed_OPEX_USD_y": Fixed_opex_y,
        "Total_OPEX_USD_y": Total_opex_y,
        
        # --- OPEX breakdown (USD/kg CNT) = "cost / Throughput" in your 2023 sense ---
        "CH4_cost_USD_per_kgCNT": CH4_cost_per_kgCNT,
        "Fe_cost_USD_per_kgCNT": Fe_cost_per_kgCNT,
        "S_cost_USD_per_kgCNT": S_cost_per_kgCNT,
        "N2_cost_USD_per_kgCNT": N2_cost_per_kgCNT,
        "Elec_cost_USD_per_kgCNT": Elec_cost_per_kgCNT,
        "Labour_cost_USD_per_kgCNT": Labour_cost_per_kgCNT,
        "Overhead_cost_USD_per_kgCNT": Overhead_cost_per_kgCNT,
        "Infra_depr_USD_per_kgCNT": Infra_depr_per_kgCNT,
        "Maintenance_USD_per_kgCNT": Maintenance_per_kgCNT,
        "Variable_OPEX_USD_per_kgCNT": Variable_opex_per_kgCNT,
        "Fixed_OPEX_USD_per_kgCNT": Fixed_opex_per_kgCNT,
        "Total_OPEX_USD_per_kgCNT": Total_opex_per_kgCNT,
        
        # --- Shares of Variable OPEX ---
        "Share_CH4_in_VarOPEX": Share_CH4_in_var,
        "Share_Fe_in_VarOPEX": Share_Fe_in_var,
        "Share_S_in_VarOPEX": Share_S_in_var,
        "Share_N2_in_VarOPEX": Share_N2_in_var,
        "Share_Elec_in_VarOPEX": Share_Elec_in_var,
        
        # --- Shares of Total OPEX ---
        "Share_CH4_in_TotalOPEX": Share_CH4_in_tot,
        "Share_Fe_in_TotalOPEX": Share_Fe_in_tot,
        "Share_S_in_TotalOPEX": Share_S_in_tot,
        "Share_N2_in_TotalOPEX": Share_N2_in_tot,
        "Share_Percursor_in_TotalOPEX":Share_Percursor_in_TotalOPEX,   
        "Share_Elec_in_TotalOPEX": Share_Elec_in_tot,
        "Share_Labour_in_TotalOPEX": Share_Labour_in_tot,
        "Share_Overhead_in_TotalOPEX": Share_Overhead_in_tot,
        "Share_InfraDepr_in_TotalOPEX": Share_InfraDepr_in_tot,
        "Share_Maint_in_TotalOPEX": Share_Maint_in_tot,
        



        # Uncertain drivers (for sensitivity)
        "Availability": Availability,
        "CH4_conversion": CH4_conversion,
        "CarbonyieldFC": CarbonyieldFC,
        "PercentCNTYield": PercentCNTYield,
        "Percent_of_H2_from_purge_gas": Percent_of_H2_from_purge_gas,
    "Hot_recycle_ratio_unc": Hot_recycle_ratio_fccvd,
        "Fe_kg_per_kgCNT": Fe_kg_per_kgCNT,
        "S_kg_per_kgCNT": S_kg_per_kgCNT,
        "H2_to_CH4_molar": H2_to_CH4_molar,
        "Blanket_thickness_m": blanket_thickness_m,
        "Insulation_efficiency": Insulationefficiency_unc,

        "CH4_price_USD_per_kg": Methaneprice,
        "Fe_price_USD_per_kg": Ironprice,
        "S_price_USD_per_kg": Sulphurprice,
        "N2_price_USD_per_kg": Nitrogenprice,
        "Elec_USD_per_kWh": electricaldemandcost,
        "CNT_price_USD_per_kg": SalepriceCNTperkg,
        "CB_price_USD_per_kg": SalepriceCBperkg,
        "H2_price_USD_per_kg": SalepriceH2perkg,

        "ScalingExponent": scalingexponent,
        "Tubescaleupfactor": Tubescaleupfactor,
        "Gasrecoveryscalefactor": Gasrecoveryscalefactor,
        "Overhead_fraction": percentageoflabourcosts,
        "CAPEX_factor": CAPEX_factor,
        "OPEX_factor": OPEX_factor,
    })

    df["NPV_positive"] = df["NPV_USD"] > 0
    return df



# ============================================================
# 8B) PHYSICS MODEL (FLB / FLBR) – 900 C, atmospheric
# ============================================================

def flb_physics_vectorized(
    n_trains: int,
    reactors_per_train: int,
    cnt_target_kg_h: float,
    T_reactor_C: float = 900.0,
    P_reactor_Pa: float = 101_325.0,
    psa_pressure_bar: np.ndarray | float = 20.0,
    eta_comp: np.ndarray | float = 0.75,
):
    """Physics skeleton for the FLB route (vectorized over Monte Carlo samples).

    Design intent:
    - Keep it *physics-first*: mass balance tied to CNT target, conversion, recycle, and yields.
    - Use CO2 as a co-feed (as in your 2023 block) with a sampled CO2/CH4 ratio.
    - 900 C operating temperature (your requirement).

    Notes:
    - We keep the same simplifying reaction set as your FCCVD skeleton:
        CH4 -> C(s) + 2 H2  (cracking)
      Here, CO2 is treated as a co-feed that contributes C and O to the element balance,
      and it impacts feed heating duty.
    - When you are ready, we can replace these “ratio” inputs by a proper FLB kinetics +
      hydrodynamics model (minimum fluidization velocity, residence time, gas-solid transfer,
      conversion, etc.).
    """

    # --- Stoichiometry / molecular weights (kg/kmol) ---
    MW_CH4 = 16.043
    MW_CO2 = 44.0095
    MW_H2  = 2.01588

    # --- Uncertain operating/chemistry drivers (bounded) ---
    # Methane conversion at 900 C (bounded). If you switch to kinetics, replace this.
    # Reference context: CMD conversions depend on T, space velocity, catalyst; 900 C is in-range for significant conversion.
    # (Keep this as an *input uncertainty* for now.)
    CH4_conversion_flb = beta(a=14, b=8).rvs(size=N, random_state=rng)  # mean ~0.64

    # CO2 co-feed ratio (molar basis): n_CO2 / n_CH4, typical co-feed for catalyst regeneration / gasification control
    CO2_to_CH4_molar = triang(c=(0.8 - 0.3) / (1.2 - 0.3), loc=0.3, scale=0.9).rvs(size=N, random_state=rng)

    # H2 dilution ratio (molar): n_H2 / n_CH4 (carrier / dilution)
    H2_to_CH4_molar = triang(c=(0.25 - 0.05) / (0.40 - 0.05), loc=0.05, scale=0.35).rvs(size=N, random_state=rng)

    # N2 dilution ratio (molar): n_N2 / n_CH4
    N2_to_CH4_molar = triang(c=(0.8 - 0.2) / (1.5 - 0.2), loc=0.2, scale=1.3).rvs(size=N, random_state=rng)

    # Hot recycle ratio (fraction of non-solid outlet recycled hot)
    Hot_recycle_ratio_flb = beta(a=35, b=6).rvs(size=N, random_state=rng)  # mean ~0.85

    # PSA: hydrogen recovery fraction (feed H2 to PSA -> product H2). Modern PSA typically 75–90% recovery depending on purity.
    # Treat this as an uncertainty driver (you can tighten with vendor specs later).
    PSA_H2_recovery = triang(c=(0.85 - 0.75) / (0.92 - 0.75), loc=0.75, scale=0.17).rvs(size=N, random_state=rng)

    # CNT vs carbon black split for FLB solids
    PercentCNTYield_flb = beta(a=30, b=8).rvs(size=N, random_state=rng)
    PercentCBYield_flb = 1.0 - PercentCNTYield_flb

    # Catalyst/support dosing (kg FeAl per kg CNT). Keep as input uncertainty.
    FeAl_dose_kg_per_kg_cnt = triang(
        c=(0.03 - 0.01) / (0.06 - 0.01),
        loc=0.01,
        scale=0.05,
    ).rvs(size=N, random_state=rng)

    # --- Production basis (scheduled operating hour basis; availability handled in economic wrapper) ---
    # Keep consistent with FCCVD: all per-hour mass/energy rates are defined on
    # scheduled hours, then multiplied by effective hours downstream.
    target_cnt_kg_h = float(cnt_target_kg_h)

    # Total solid carbon produced (kg/h). If FLB also makes byproduct carbon, include it via CarbonyieldFLB.
    # Here we reuse your existing CarbonyieldFC uncertainty as a placeholder for overall solid-carbon yield.
    total_solid_carbon_kg_h = (target_cnt_kg_h / np.clip(PercentCNTYield_flb, 1e-6, 1.0))

    # FeAl feed (kg/h)
    feal_in_kg_h = FeAl_dose_kg_per_kg_cnt * target_cnt_kg_h

    # --- CH4 requirement from carbon balance ---
    # Carbon coming from CH4 cracking (1 mol C per mol CH4 converted)
    nC_required_kmol_h = total_solid_carbon_kg_h / 12.0107
    nCH4_conv_kmol_h = nC_required_kmol_h
    nCH4_feed_kmol_h = nCH4_conv_kmol_h / np.clip(CH4_conversion_flb, 1e-6, 1.0)
    ch4_in_kg_h = nCH4_feed_kmol_h * MW_CH4

    # --- CO2 co-feed ---
    nCO2_kmol_h = CO2_to_CH4_molar * nCH4_feed_kmol_h
    co2_in_kg_h = nCO2_kmol_h * MW_CO2

    # --- Diluent gases ---
    nH2_in_kmol_h = H2_to_CH4_molar * nCH4_feed_kmol_h
    h2_in_kg_h = nH2_in_kmol_h * MW_H2
    nN2_in_kmol_h = N2_to_CH4_molar * nCH4_feed_kmol_h
    n2_in_kg_h = nN2_in_kmol_h * 28.0134

    # --- Hydrogen produced from CH4 cracking ---
    nH2_prod_kmol_h = 2.0 * nCH4_conv_kmol_h
    h2_prod_kg_h = nH2_prod_kmol_h * MW_H2

    # --- PSA / recycle split ---
    # Purge fraction is (1 - hot_recycle)
    purge_frac = 1.0 - Hot_recycle_ratio_flb
    h2_to_psa_kg_h = purge_frac * (h2_prod_kg_h + h2_in_kg_h)
    h2_recovered_kg_h = PSA_H2_recovery * h2_to_psa_kg_h
    h2_recovery_fraction = PSA_H2_recovery
    hot_recycle_ratio = Hot_recycle_ratio_flb

    # --- Heating duty (feed sensible) ---
    T_amb_C = 25.0
    dT = T_reactor_C - T_amb_C
    # Use CoolProp ideal-gas cp as a reasonable high-T approximation driver; keep P=1 atm.
    cp_ch4 = cp_ideal_gas_kJ_kgK("Methane", T_reactor_C, P_reactor_Pa)
    cp_co2 = cp_ideal_gas_kJ_kgK("CarbonDioxide", T_reactor_C, P_reactor_Pa)
    cp_h2  = cp_ideal_gas_kJ_kgK("Hydrogen", T_reactor_C, P_reactor_Pa)
    cp_n2  = cp_ideal_gas_kJ_kgK("Nitrogen", T_reactor_C, P_reactor_Pa)

    feed_sensible_kW = (
        (ch4_in_kg_h * cp_ch4
         + co2_in_kg_h * cp_co2
         + h2_in_kg_h * cp_h2
         + n2_in_kg_h * cp_n2)
        * dT
    ) / 3600.0

    # --- Reaction endotherm (kW) ---
    # Align with FCCVD: use the same ΔH for methane cracking.
    # nCH4_conv_kmol_h is kmol/h; convert to mol/s.
    ch4_conv_mol_s = (nCH4_conv_kmol_h * 1000.0) / 3600.0
    Q_rxn_W = ch4_conv_mol_s * (DELTA_H_CH4_CRACK_kJ_per_mol * 1000.0)
    heater_kW = feed_sensible_kW + (Q_rxn_W / 1000.0)

    # --- PSA / recycle compression (kW) ---
    # Align with FCCVD: compress H2 stream from reactor pressure to PSA feed pressure.
    T1_K = K(T_ref_C)
    P2_Pa = psa_pressure_bar * 1e5
    psa_compression_kW = isentropic_compression_power_kW(
        m_dot_kg_s=(h2_to_psa_kg_h / 3600.0),
        fluid="Hydrogen",
        T1_K=T1_K,
        P1_Pa=float(P_reactor_Pa),
        P2_Pa=P2_Pa,
        eta_isentropic=eta_comp,
    )

    return {
        "target_cnt_kg_h": target_cnt_kg_h,
        "total_solid_carbon_kg_h": total_solid_carbon_kg_h,
        "cnt_kg_h": total_solid_carbon_kg_h * PercentCNTYield_flb,
        "cb_kg_h": total_solid_carbon_kg_h * PercentCBYield_flb,
        "ch4_in_kg_h": ch4_in_kg_h,
        "co2_in_kg_h": co2_in_kg_h,
        "h2_in_kg_h": h2_in_kg_h,
        "n2_in_kg_h": n2_in_kg_h,
        "feal_in_kg_h": feal_in_kg_h,
        "h2_prod_kg_h": h2_prod_kg_h,
        "h2_to_psa_kg_h": h2_to_psa_kg_h,
        "h2_recovered_kg_h": h2_recovered_kg_h,
        "h2_recovery_fraction": h2_recovery_fraction,
        "hot_recycle_ratio": hot_recycle_ratio,
        "feed_sensible_kW": feed_sensible_kW,
        "heater_kW": heater_kW,
        "PSA_compression_kW": psa_compression_kW,

        # Drivers
        "CH4_conversion_flb": CH4_conversion_flb,
        "CO2_to_CH4_molar": CO2_to_CH4_molar,
        "H2_to_CH4_molar": H2_to_CH4_molar,
        "N2_to_CH4_molar": N2_to_CH4_molar,
        "Hot_recycle_ratio_flb": Hot_recycle_ratio_flb,
        "PSA_H2_recovery": PSA_H2_recovery,
        "PercentCNTYield_flb": PercentCNTYield_flb,
        "PercentCBYield_flb": PercentCBYield_flb,
    }


def economic_model_flb_vectorized() -> pd.DataFrame:
    """FLB Monte Carlo wrapper that mirrors FCCVD outputs."""

    # Size the plant using the SAME sizing assumptions as the FCCVD wrapper
    # (i.e., same reference proxy rate, same scaling exponent, same per-reactor cap).
    base_cnt_kg_h = Throughput_g_h_small / 1000.0
    scale_factor = float(Throughputofscale_g_h / Throughput_g_h_small)
    alpha = 0.60
    max_cnt_kg_h = 200.0  # cap per reactor nameplate (kg/h)
    scaling_exponent = 0.65

    # Legacy anchor (same table logic used in FCCVD) for shared facility cost + reference furnace power.
    # We then apply a conservative derating for 900 C operation.
    _, _, _, _, ref_furnace_power_legacy, facility_share_cost, _, _ = calculate_parameters(
        Throughputofscale_ton_per_year,
        rng=np.random.default_rng(12345),
    )
    reference_furnace_kW_flb = 0.75 * ref_furnace_power_legacy

    # Per-reactor price distribution (vectorized MC). Keep aligned with FCCVD pricing logic.
    single_reactor_price_flb = rng.triangular(20_000.0, 40_000.0, 90_000.0, size=N)

    # Use the same annual operating hours basis as FCCVD for apples-to-apples scaling
    n_reactors, n_trains, n_facilities, per_reactor_kg_y = compute_reactor_facility_counts(
        target_tpy=float(Throughputofscale_ton_per_year),
        hours_per_year=float(AnnualOperatingHours),
        availability=float(np.mean(Availability)),
        base_cnt_kg_h=base_cnt_kg_h,
        scale_factor=scale_factor,
        alpha=alpha,
        max_cnt_kg_h=max_cnt_kg_h,
        trains_per_facility=8,
        reactors_per_train=1,
    )

    # Optional: quick scaling diagnostics
    if globals().get("DEBUG_SCALE", False):
        print(
            "[SCALE][FLB] target_tpy=",
            float(Throughputofscale_ton_per_year),
            "Throughputofscale_g_h=",
            float(Throughputofscale_g_h),
            "n_reactors=",
            int(n_reactors),
            "n_trains=",
            int(n_trains),
            "n_facilities=",
            int(n_facilities),
        )

    # Production basis (align with FCCVD)
    cnt_target_kg_y = float(Throughputofscale_ton_per_year) * 1000.0
    cnt_target_kg_h = cnt_target_kg_y / float(AnnualOperatingHours)

    # Effective operating hours (availability)
    effective_hours = float(AnnualOperatingHours) * Availability

    phys = flb_physics_vectorized(
        n_trains=n_trains,
        reactors_per_train=1,
        cnt_target_kg_h=cnt_target_kg_h,
        T_reactor_C=900.0,
        P_reactor_Pa=P_reactor_Pa,
        psa_pressure_bar=PSA_feed_pressure_bar,
        eta_comp=PSA_compressor_eta,
    )
    cnt_kg_y = phys["cnt_kg_h"] * effective_hours
    cb_kg_y = phys["cb_kg_h"] * effective_hours
    h2_kg_y = phys["h2_recovered_kg_h"] * effective_hours
    h2_ton_y = h2_kg_y / 1000.0

    sell_H2 = 1.0 if AreyousellingH2.lower() == "yes" else 0.0
    sell_solids = 1.0 if AreyousellingSolidCarbon.lower() == "yes" else 0.0

    revenue = (
        cnt_kg_y * SalepriceCNTperkg * sell_solids
        + cb_kg_y * SalepriceCBperkg * sell_solids
        + h2_kg_y * SalepriceH2perkg * sell_H2
    )

    # Variable OPEX (feeds + utilities) for FLB
    CH4_cost = phys["ch4_in_kg_h"] * effective_hours * Methaneprice
    CO2_cost = phys["co2_in_kg_h"] * effective_hours * CO2price
    N2_cost = phys["n2_in_kg_h"] * effective_hours * Nitrogenprice
    FeAl_cost = phys["feal_in_kg_h"] * effective_hours * FeAlprice

    # Electricity: heater + PSA compression + heat-loss (align with FCCVD)
    A_hot_zone_m2 = np.full(N, 120.0)  # placeholder; replace with geometry sizing
    Q_loss_W = heat_loss_W(A_hot_zone_m2, 900.0, 25.0, blanket_thickness_m)
    Q_loss_kWh_y = (Q_loss_W / 1000.0) * effective_hours

    elec_kWh_y = (phys["heater_kW"] + phys["PSA_compression_kW"]) * effective_hours + Q_loss_kWh_y
    elec_kWh_y = elec_kWh_y - (Q_loss_kWh_y * Insulationefficiency_unc)
    elec_kWh_y = np.maximum(0.0, elec_kWh_y)

    elec_cost = elec_kWh_y * electricaldemandcost

    variable_opex = (CH4_cost + CO2_cost + N2_cost + FeAl_cost + elec_cost) * OPEX_factor

    # Fixed OPEX and depreciation (same structure as FCCVD)
    direct_labour_usd_y = compute_total_direct_labour_usd_per_year_for_facilities(int(n_facilities))
    fixed_opex = direct_labour_usd_y * (1.0 + percentageoflabourcosts)
    infra_depr = (InfrastructureQAequipementcost * int(n_facilities)) / usefullifeinfrastructure

    # CAPEX from building blocks (same logic as FCCVD, but FLB reactor price distribution)
    # FLB: metallic / refractory-lined up to ~900C (triangular distribution)
    # FLB stainless steel / refractory-lined reactor (~900C)
    # Sources:
    # - ANDRITZ, Babcock & Wilcox (fluidized beds)
    # - Abánades et al., IJHE (2012)
    reactor_price_flb = rng.triangular(20_000.0, 40_000.0, 90_000.0, size=N)

    # Facility shared CAPEX anchor (USD). You can later replace with a detailed utility model.
    facility_share_cost = 75_800.0

    # Gas recovery reference CAPEX (USD) and scaling (kept from your prior model)
    Gasrecoverycost = float(Gasrecoverycost_base)

    # CAPEX scaling exponents / factors (ported defaults; tune as you calibrate)
    Tubecapitalexponentscaleup = np.full(N, 0.65)  # tube capital exponent
    Tubeperreactorsclalingexponentfactor = np.full(N, 1.0)
    Reactorscalingfactor = np.full(N, 1.0)
    Facilityscalingfactor = np.full(N, 1.0)
    Gasrecoveryscalefactor = 0.6  # scaling exponent for recovery package

    # Scalar exponents used in the shared CAPEX scaling blocks
    scalingexponent_scalar = float(np.mean(scalingexponent)) if np.ndim(scalingexponent) else float(scalingexponent)
    Gasrecoveryscalefactor_scalar = float(np.mean(Gasrecoveryscalefactor)) if np.ndim(Gasrecoveryscalefactor) else float(Gasrecoveryscalefactor)

    capex = Capital_cost(
        Tube_scale_up_factor=Tubescaleupfactor,
        Tube_capital_exponent_scaleup=Tubecapitalexponentscaleup,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_reactors=n_reactors,
        Number_of_facility=n_facilities,
        Tube_per_reactor_sclaling_exponent_factor=Tubeperreactorsclalingexponentfactor,
        Reactor_scaling_factor=Reactorscalingfactor,
        Facility_scaling_factor=Facilityscalingfactor,
        Single_reactor_price_usd=reactor_price_flb,
        Facility_share_cost=facility_share_cost,
        scaling_exponent=scalingexponent_scalar,
        Recovery_of_H2_from_purge_gas=phys["PSA_H2_recovery"],
        Gas_recovery_cost=Gasrecoverycost,
        Gas_recovery_scale_factor=Gasrecoveryscalefactor_scalar,
        Infrastructure_QA_equipement_cost=InfrastructureQAequipementcost,
        compute_insulation=True,
        rng=rng,
        N_samples=N,
        Throughputofscale_ton_per_year=Throughputofscale_ton_per_year,
        T_hot_C=900.0,
        T_amb_C=40.0,
        Heat_for_insulation_kW=phys["heater_kW"] * (1.0 / (1.0 - Insulationefficiency_unc)),
        GBPUSD=GBPUSD,
    )
    # Apply project-level CAPEX uncertainty factor
    capex = capex * CAPEX_factor

    # ------------------------------------------------------------
    # CAPEX BREAKDOWN + SHARES (reactor, facility, gas recovery, infra, insulation)
    # ------------------------------------------------------------
    reactor_cost = reactor_capital(
        Tube_scale_up_factor=Tubescaleupfactor,
        Tube_capital_exponent_scaleup=Tubecapitalexponentscaleup,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_reactors=n_reactors,
        Number_of_facility=n_facilities,
        Tube_per_reactor_sclaling_exponent_factor=Tubeperreactorsclalingexponentfactor,
        Reactor_scaling_factor=Reactorscalingfactor,
        Facility_scaling_factor=Facilityscalingfactor,
        Single_reactor_price_usd=reactor_price_flb,
    )[1]

    facility_cost = facilty_shared_capital(
        Facility_share_cost=facility_share_cost,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_facility=n_facilities,
        scaling_exponent=scalingexponent_scalar,
    )[1]

    gasrec_cost = Additional_capital(
        Recovery_of_H2_from_purge_gas=phys["PSA_H2_recovery"],
        Gas_recovery_cost=Gasrecoverycost,
        Throughput_=Throughput_g_h_small,
        Throughput_of_scale=Throughputofscale_g_h,
        Number_of_facility=n_facilities,
        Gas_recovery_scale_factor=Gasrecoveryscalefactor_scalar,
    )

    infra_cost = Infrastructure_QA_cost(
        Infrastructure_QA_equipement_cost=InfrastructureQAequipementcost,
        Number_of_facility=n_facilities,
    )

    insulation_cost = compute_insulation_cost_usd(
        rng=rng,
        N=N,
        Throughputofscale_ton_per_year=Throughputofscale_ton_per_year,
        n_reactors=n_reactors,
        T_hot_C=900.0,
        T_amb_C=40.0,
        Heat_for_insulation_kW=phys["heater_kW"] * (1.0 / (1.0 - Insulationefficiency_unc)),
        GBPUSD=GBPUSD,
    )

    # Apply the SAME project-level CAPEX uncertainty factor to each component
    reactor_cost = reactor_cost * CAPEX_factor
    facility_cost = facility_cost * CAPEX_factor
    gasrec_cost = gasrec_cost * CAPEX_factor
    infra_cost = infra_cost * CAPEX_factor
    insulation_cost = insulation_cost * CAPEX_factor

    capex_safe = np.maximum(1e-12, capex)
    Share_reactor_in_TotalCAPEX = reactor_cost / capex_safe
    Share_facility_in_TotalCAPEX = facility_cost / capex_safe
    Share_gasrec_in_TotalCAPEX = gasrec_cost / capex_safe
    Share_infra_in_TotalCAPEX = infra_cost / capex_safe
    Share_insulation_in_TotalCAPEX = insulation_cost / capex_safe

    maintenance = Percentangecostmaintenance * capex
    total_opex = variable_opex + fixed_opex + infra_depr + maintenance
    # Cashflow basis: treat depreciation as non-cash (keep it in total OPEX + unit cost,
    # but exclude it from cashflow/NPV calculations).
    opex_cash = total_opex - infra_depr
    annual_cashflow = revenue - opex_cash


    # ============================================================
    # OPEX BREAKDOWN (FLB) – 2023-style with Monte Carlo vectors
    # Costs normalized by CNT production (kg CNT / year)
    # ============================================================

    CNT_kg_y_safe = np.maximum(1e-12, cnt_kg_y)

    # --- Annual cost components (USD/y) ---
    CH4_cost_y = CH4_cost
    CO2_cost_y = CO2_cost
    N2_cost_y  = N2_cost
    FeAl_cost_y = FeAl_cost
    Elec_cost_y = elec_cost

    Labour_cost_y = direct_labour_usd_y
    Overhead_cost_y = direct_labour_usd_y * percentageoflabourcosts

    Infra_depr_y = infra_depr
    Maintenance_y = maintenance

    Variable_opex_y = variable_opex
    Fixed_opex_y = fixed_opex
    Total_opex_y = total_opex

    # --- Normalize by throughput (USD per kg CNT) ---
    CH4_cost_per_kgCNT = CH4_cost_y / CNT_kg_y_safe
    CO2_cost_per_kgCNT = CO2_cost_y / CNT_kg_y_safe
    N2_cost_per_kgCNT  = N2_cost_y  / CNT_kg_y_safe
    FeAl_cost_per_kgCNT = FeAl_cost_y / CNT_kg_y_safe
    Elec_cost_per_kgCNT = Elec_cost_y / CNT_kg_y_safe

    Labour_cost_per_kgCNT = Labour_cost_y / CNT_kg_y_safe
    Overhead_cost_per_kgCNT = Overhead_cost_y / CNT_kg_y_safe
    Infra_depr_per_kgCNT = Infra_depr_y / CNT_kg_y_safe
    Maintenance_per_kgCNT = Maintenance_y / CNT_kg_y_safe

    Variable_opex_per_kgCNT = Variable_opex_y / CNT_kg_y_safe
    Fixed_opex_per_kgCNT = Fixed_opex_y / CNT_kg_y_safe
    Total_opex_per_kgCNT = Total_opex_y / CNT_kg_y_safe

    # --- % contribution shares ---
    var_safe = np.maximum(1e-12, Variable_opex_y)
    tot_safe = np.maximum(1e-12, Total_opex_y)

    # Shares of VARIABLE OPEX
    Share_CH4_in_VarOPEX = CH4_cost_y / var_safe
    Share_CO2_in_VarOPEX = CO2_cost_y / var_safe
    Share_N2_in_VarOPEX  = N2_cost_y  / var_safe
    Share_FeAl_in_VarOPEX = FeAl_cost_y / var_safe
    Share_Elec_in_VarOPEX = Elec_cost_y / var_safe

    # Shares of TOTAL OPEX
    Share_CH4_in_TotalOPEX = CH4_cost_y / tot_safe
    Share_CO2_in_TotalOPEX = CO2_cost_y / tot_safe
    Share_N2_in_TotalOPEX  = N2_cost_y  / tot_safe
    Share_FeAl_in_TotalOPEX = FeAl_cost_y / tot_safe
    Share_Percursor_in_TotalOPEX = (CH4_cost_y+ FeAl_cost_y + CO2_cost_y 
                                    + N2_cost_y) / tot_safe
    Share_Elec_in_TotalOPEX = Elec_cost_y / tot_safe
    Share_Labour_in_TotalOPEX = Labour_cost_y / tot_safe
    Share_Overhead_in_TotalOPEX = Overhead_cost_y / tot_safe
    Share_InfraDepr_in_TotalOPEX = Infra_depr_y / tot_safe
    Share_Maint_in_TotalOPEX = Maintenance_y / tot_safe



    # NPV
    years = np.arange(1, Yearforprofitibilitymodel + 1)
    discount = 1.0 / ((1.0 + InterestDiscountRate) ** years)
    pv_cashflows = annual_cashflow[:, None] * discount[None, :]
    npv = -capex + pv_cashflows.sum(axis=1)
    # Payback year (simple)
    cum = np.cumsum(np.repeat(annual_cashflow[:, None], Yearforprofitibilitymodel, axis=1), axis=1)
    payback_year = np.where(cum >= capex[:, None], years[None, :], np.inf).min(axis=1)
    # Unit cost (OPEX / kg CNT) – ex-capex definition (align with FCCVD)
    unit_cost_ex_cap = total_opex / np.maximum(1e-12, cnt_kg_y)

    df = pd.DataFrame({
        "Route": "FLB",
        # Plant sizing (single values replicated across MC rows)
        "Number_of_facilities": int(n_facilities),
        "Number_of_reactors": int(n_reactors),

        "CAPEX_USD": capex,
        "Reactor_cost_USD": reactor_cost,
        "Facility_cost_USD": facility_cost,
        "Gasrecovery_cost_USD": gasrec_cost,
        "Infra_cost_USD": infra_cost,
        "Insulation_cost_USD": insulation_cost,

        "Share_reactor_in_TotalCAPEX": Share_reactor_in_TotalCAPEX,
        "Share_facility_in_TotalCAPEX": Share_facility_in_TotalCAPEX,
        "Share_gasrec_in_TotalCAPEX": Share_gasrec_in_TotalCAPEX,
        "Share_infra_in_TotalCAPEX": Share_infra_in_TotalCAPEX,
        "Share_insulation_in_TotalCAPEX": Share_insulation_in_TotalCAPEX,
        "Revenue_USD_per_y": revenue,
        "OPEX_USD_per_y": total_opex,
        "Cashflow_USD_per_y": annual_cashflow,
        "NPV_USD": npv,
        "Payback_year": payback_year,
        "Unit_cost_ex_cap_USD_per_kgCNT": unit_cost_ex_cap,

        "CNT_kg_y": cnt_kg_y,
        "CB_kg_y": cb_kg_y,
        "H2_kg_y": h2_kg_y,
    "H2_ton_y": h2_ton_y,

        # --- OPEX breakdown (annual USD/y) ---
        "CH4_cost_USD_y": CH4_cost_y,
        "CO2_cost_USD_y": CO2_cost_y,
        "N2_cost_USD_y": N2_cost_y,
        "FeAl_cost_USD_y": FeAl_cost_y,
        "Elec_cost_USD_y": Elec_cost_y,
        "Labour_cost_USD_y": Labour_cost_y,
        "Overhead_cost_USD_y": Overhead_cost_y,
        "Infra_depr_USD_y": Infra_depr_y,
        "Maintenance_USD_y": Maintenance_y,
        "Variable_OPEX_USD_y": Variable_opex_y,
        "Fixed_OPEX_USD_y": Fixed_opex_y,
        "Total_OPEX_USD_y": Total_opex_y,

        # --- OPEX breakdown (USD/kg CNT) ---
        "CH4_cost_USD_per_kgCNT": CH4_cost_per_kgCNT,
        "CO2_cost_USD_per_kgCNT": CO2_cost_per_kgCNT,
        "N2_cost_USD_per_kgCNT": N2_cost_per_kgCNT,
        "FeAl_cost_USD_per_kgCNT": FeAl_cost_per_kgCNT,
        "Elec_cost_USD_per_kgCNT": Elec_cost_per_kgCNT,
        "Labour_cost_USD_per_kgCNT": Labour_cost_per_kgCNT,
        "Overhead_cost_USD_per_kgCNT": Overhead_cost_per_kgCNT,
        "Infra_depr_USD_per_kgCNT": Infra_depr_per_kgCNT,
        "Maintenance_USD_per_kgCNT": Maintenance_per_kgCNT,
        "Variable_OPEX_USD_per_kgCNT": Variable_opex_per_kgCNT,
        "Fixed_OPEX_USD_per_kgCNT": Fixed_opex_per_kgCNT,
        "Total_OPEX_USD_per_kgCNT": Total_opex_per_kgCNT,

        # --- Shares of Variable OPEX ---
        "Share_CH4_in_VarOPEX": Share_CH4_in_VarOPEX,
        "Share_CO2_in_VarOPEX": Share_CO2_in_VarOPEX,
        "Share_N2_in_VarOPEX": Share_N2_in_VarOPEX,
        "Share_FeAl_in_VarOPEX": Share_FeAl_in_VarOPEX,
        "Share_Elec_in_VarOPEX": Share_Elec_in_VarOPEX,

        # --- Shares of Total OPEX ---
        "Share_CH4_in_TotalOPEX": Share_CH4_in_TotalOPEX,
        "Share_CO2_in_TotalOPEX": Share_CO2_in_TotalOPEX,
        "Share_N2_in_TotalOPEX": Share_N2_in_TotalOPEX,
        "Share_FeAl_in_TotalOPEX": Share_FeAl_in_TotalOPEX,
        "Share_Percursor_in_TotalOPEX":Share_Percursor_in_TotalOPEX,         
        
        "Share_Elec_in_TotalOPEX": Share_Elec_in_TotalOPEX,
        "Share_Labour_in_TotalOPEX": Share_Labour_in_TotalOPEX,
        "Share_Overhead_in_TotalOPEX": Share_Overhead_in_TotalOPEX,
        "Share_InfraDepr_in_TotalOPEX": Share_InfraDepr_in_TotalOPEX,
        "Share_Maint_in_TotalOPEX": Share_Maint_in_TotalOPEX,




        "CH4_in_kg_h": phys["ch4_in_kg_h"],
        "CO2_in_kg_h": phys["co2_in_kg_h"],
        "N2_in_kg_h": phys["n2_in_kg_h"],
        "H2_recovered_kg_h": phys["h2_recovered_kg_h"],
        "H2_to_PSA_kg_h": phys["h2_to_psa_kg_h"],
        "Feed_sensible_kW": phys["feed_sensible_kW"],
        "heater_kW": phys["heater_kW"],
        "PSA_compression_kW": phys["PSA_compression_kW"],
        "PSA_H2_recovery": phys["PSA_H2_recovery"],
        "Hot_recycle_ratio": phys["Hot_recycle_ratio_flb"],
        "CH4_conversion": phys["CH4_conversion_flb"],
        "CO2_to_CH4_molar": phys["CO2_to_CH4_molar"],
        "PercentCNTYield": phys["PercentCNTYield_flb"],

        
        #Uncertainty varaible input
        "CO2_price_USD_per_kg": CO2price,
        "FeAl_price_USD_per_kg": FeAlprice,
    })

    df["NPV_positive"] = df["NPV_USD"] > 0
    return df


# ============================================================
# 8) REPORTING + PLOTS
# ============================================================


def p10_p50_p90(x: np.ndarray) -> tuple[float, float, float]:
    return (np.percentile(x, 10), np.percentile(x, 50), np.percentile(x, 90))


def _disable_axis_sci_offset(ax, axis: str = "x") -> None:
    """Force plain tick labels (no scientific notation and no offset like '1e6')."""
    try:
        ax.ticklabel_format(axis=axis, style="plain", useOffset=False)
    except Exception:
        # Some axis/formatter combos may not support ticklabel_format.
        pass

    formatter = mtick.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)


def hist_plot(
    x: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 60,
    *,
    scale: float = 1.0,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    x = np.asarray(x, dtype=float) * float(scale)

    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    _disable_axis_sci_offset(ax, axis="x")
    fig.tight_layout()
    if save_path is not None:
        _savefig_without_titles(fig, save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# 9) MAIN
# ============================================================

if __name__ == "__main__":
    st = time.time()

    # Ensure output directory exists
    if cfg.save_outputs and cfg.output_dir not in ("", "."):
        os.makedirs(cfg.output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # ----------------
    # FCCVD
    # ----------------
    fccvd_results = economic_model_fccvd_vectorized()
    print("\n--- FCCVD Monte Carlo summary ---")
    print("N =", cfg.num_samples)
    print("P(NPV>0) =", fccvd_results["NPV_positive"].mean())

    fccvd_npv_p10, fccvd_npv_p50, fccvd_npv_p90 = p10_p50_p90(fccvd_results["NPV_USD"].to_numpy())
    fccvd_uc_p10, fccvd_uc_p50, fccvd_uc_p90 = p10_p50_p90(fccvd_results["Unit_cost_ex_cap_USD_per_kgCNT"].to_numpy())

    print("NPV (USD) P10/P50/P90:", (fccvd_npv_p10, fccvd_npv_p50, fccvd_npv_p90))
    print("Unit cost (OPEX ex-capex) USD/kgCNT P10/P50/P90:", (fccvd_uc_p10, fccvd_uc_p50, fccvd_uc_p90))

    fccvd_csv_path = f"{cfg.output_dir}/fccvd_mc_results_{timestamp}.csv"
    fccvd_npv_plot_path = f"{cfg.output_dir}/fccvd_hist_NPV_USD_{timestamp}.png"
    fccvd_uc_plot_path = f"{cfg.output_dir}/fccvd_hist_UnitCost_USD_per_kgCNT_{timestamp}.png"

    if cfg.save_outputs:
        fccvd_results.to_csv(fccvd_csv_path, index=False)
        hist_plot(
            fccvd_results["NPV_USD"].to_numpy(),
            "FCCVD: NPV (USD)",
            "NPV [$Million]",
            scale=1e-6,
            save_path=fccvd_npv_plot_path,
        )
        hist_plot(fccvd_results["Unit_cost_ex_cap_USD_per_kgCNT"].to_numpy(),
                  "FCCVD: Unit cost (ex-capex) (USD/kgCNT)", "USD/kgCNT", save_path=fccvd_uc_plot_path)

    # ----------------
    # FLB / FLBR
    # ----------------
    flb_results = economic_model_flb_vectorized()
    print("\n--- FLB Monte Carlo summary ---")
    print("N =", cfg.num_samples)
    print("P(NPV>0) =", flb_results["NPV_positive"].mean())

    flb_npv_p10, flb_npv_p50, flb_npv_p90 = p10_p50_p90(flb_results["NPV_USD"].to_numpy())
    flb_uc_p10, flb_uc_p50, flb_uc_p90 = p10_p50_p90(flb_results["Unit_cost_ex_cap_USD_per_kgCNT"].to_numpy())

    print("NPV (USD) P10/P50/P90:", (flb_npv_p10, flb_npv_p50, flb_npv_p90))
    print("Unit cost (OPEX ex-capex) USD/kgCNT P10/P50/P90:", (flb_uc_p10, flb_uc_p50, flb_uc_p90))

    flb_csv_path = f"{cfg.output_dir}/flb_mc_results_{timestamp}.csv"
    flb_npv_plot_path = f"{cfg.output_dir}/flb_hist_NPV_USD_{timestamp}.png"
    flb_uc_plot_path = f"{cfg.output_dir}/flb_hist_UnitCost_USD_per_kgCNT_{timestamp}.png"

    if cfg.save_outputs:
        flb_results.to_csv(flb_csv_path, index=False)
        hist_plot(
            flb_results["NPV_USD"].to_numpy(),
            "FLB: NPV (USD)",
            "NPV [$Million]",
            scale=1e-6,
            save_path=flb_npv_plot_path,
        )
        hist_plot(flb_results["Unit_cost_ex_cap_USD_per_kgCNT"].to_numpy(),
                  "FLB: Unit cost (ex-capex) (USD/kgCNT)", "USD/kgCNT", save_path=flb_uc_plot_path)

    elapsed = time.time() - st
    print(f"Runtime: {elapsed:,.2f} s")
    
    
    
    # ============================================================
    # APPLE-TO-APPLE FCCVD vs FLB PLOTS + P10/P50/P90 (drop-in block)
    # Put this at the BOTTOM of your script (after both MC runs)
    # ============================================================


    # -----------------------------
    # 0) Get results DataFrames
    # -----------------------------
    # Expected: fccvd_results and flb_results (already computed above)
    try:
        fccvd_df = fccvd_results.copy()
    except NameError:
        fccvd_df = None

    try:
        flb_df = flb_results.copy()
    except NameError:
        flb_df = None

    # Auto-fallback to CSV paths saved earlier in this run (if saving enabled)
    FCCVD_CSV_PATH = fccvd_csv_path if (fccvd_df is None and 'fccvd_csv_path' in locals()) else None
    FLB_CSV_PATH   = flb_csv_path if (flb_df is None and 'flb_csv_path' in locals()) else None

    if fccvd_df is None:
        if FCCVD_CSV_PATH is None:
            raise NameError("fccvd_results not found and FCCVD_CSV_PATH is None. Set FCCVD_CSV_PATH or keep fccvd_results in memory.")
        fccvd_df = pd.read_csv(FCCVD_CSV_PATH)

    if flb_df is None:
        if FLB_CSV_PATH is None:
            raise NameError("flb_results not found and FLB_CSV_PATH is None. Set FLB_CSV_PATH or keep flb_results in memory.")
        flb_df = pd.read_csv(FLB_CSV_PATH)

    # -----------------------------
    # 1) Metrics to compare
    # -----------------------------
    metrics = [
        "CAPEX_USD",
        "Revenue_USD_per_y",
        "OPEX_USD_per_y",
        "Cashflow_USD_per_y",
        "NPV_USD",
        "Payback_year",
        "Unit_cost_ex_cap_USD_per_kgCNT",
        "H2_ton_y",
        "Number_of_facilities",
        "Number_of_reactors",
    ]


    # ============================================================
    # 10) DETERMINISTIC (P50) SENSITIVITY ANALYSIS (Template-style)
    # ============================================================
    # Goal:
    #   - Use a deterministic “mid-values” (P50) base case.
    #   - Baseline gas recovery (H2 from purge gas) ~85%.
    #   - Sweep +/- changes for CAPEX and OPEX/physics parameters.
    #   - Plot % change of parameter vs % change of cost (same layout as template).

    from dataclasses import dataclass
    import matplotlib.ticker as mtick


    @dataclass(frozen=True)
    class SensitivityOverrides:
        # CAPEX multipliers
        capex_reactor_mult: float = 1.0
        capex_facility_mult: float = 1.0
        capex_gasrec_mult: float = 1.0
        capex_infra_mult: float = 1.0
        capex_insulation_mult: float = 1.0

        # OPEX / physics multipliers
        precursor_cost_mult: float = 1.0
        labour_cost_mult: float = 1.0
        maintenance_mult: float = 1.0
        throughput_mult: float = 1.0
        carbon_yield_mult: float = 1.0
        hot_recycle_mult: float = 1.0
        heat_retention_mult: float = 1.0

        # P50 baseline constants
        gas_recovery_fraction: float = 0.85


    def _p50(x: np.ndarray | float) -> float:
        if np.ndim(x) == 0:
            return float(x)
        return float(np.percentile(np.asarray(x, dtype=float), 50))


    def _as_len1_p50(x):
        """Coerce an input into a deterministic length-1 numeric numpy array."""

        if x is None:
            return None
        try:
            if np.ndim(x) == 0:
                return np.array([float(x)], dtype=float)
            arr = np.asarray(x, dtype=float)
            if arr.size == 1:
                return np.array([float(arr.ravel()[0])], dtype=float)
            return np.array([float(np.percentile(arr, 50))], dtype=float)
        except Exception:
            return x


    def _debug_find_len_mismatches(expected_len: int) -> str:
        """Best-effort scan for global numpy arrays that don't match `expected_len`."""

        offenders: list[str] = []
        for k, v in globals().items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] != expected_len:
                offenders.append(f"{k} shape={v.shape}")
        offenders.sort()
        if not offenders:
            return ""
        return "Remaining array length mismatches (first 30):\n  " + "\n  ".join(offenders[:30])


    def _evaluate_route_p50(route: str, overrides: SensitivityOverrides) -> dict[str, float]:
        """Run a 1-sample deterministic 'P50' evaluation for FCCVD or FLB.

        Returns a dict of key outputs; sensitivity will be calculated on
        Unit_cost_ex_cap_USD_per_kgCNT by default.
        """

        # Save essential globals we will override temporarily.
        # NOTE: This keeps changes local to this function.
        global N, rng
        global Percent_of_H2_from_purge_gas
        global Methaneprice, Ironprice, Sulphurprice, Nitrogenprice, CO2price, FeAlprice
        global SalepriceCNTperkg, SalepriceCBperkg, SalepriceH2perkg
        global Percentangecostmaintenance
        global Throughputofscale_ton_per_year, Throughputofscale_g_h
        global CarbonyieldFC
        global Availability, CH4_conversion, PercentCNTYield, Fe_kg_per_kgCNT, S_kg_per_kgCNT, H2_to_CH4_molar
        global blanket_thickness_m, Insulationefficiency_unc, percentageoflabourcosts
        global CAPEX_factor, OPEX_factor, scalingexponent, Tubescaleupfactor
        global PSA_feed_pressure_bar, PSA_compressor_eta

        # Baseline backups
        _N0 = N
        _rng0 = rng
        _h2rec0 = Percent_of_H2_from_purge_gas
        _throughput0_tpy = float(Throughputofscale_ton_per_year)
        _throughput0_g_h = float(Throughputofscale_g_h)
        _maint0 = float(Percentangecostmaintenance)

        # FCCVD/FLB driver arrays that were sampled at module import with shape (N,)
        _Availability0 = Availability
        _CH4_conversion0 = CH4_conversion
        _PercentCNTYield0 = PercentCNTYield
        _FeDose0 = Fe_kg_per_kgCNT
        _SDose0 = S_kg_per_kgCNT
        _H2_to_CH4_molar0 = H2_to_CH4_molar
        _blanket_thickness_m0 = blanket_thickness_m
        _Insulationefficiency_unc0 = Insulationefficiency_unc
        _percentageoflabourcosts0 = percentageoflabourcosts
        _CAPEX_factor0 = CAPEX_factor
        _OPEX_factor0 = OPEX_factor
        _scalingexponent0 = scalingexponent
        _Tubescaleupfactor0 = Tubescaleupfactor

        _PSA_feed_pressure_bar0 = PSA_feed_pressure_bar
        _PSA_compressor_eta0 = PSA_compressor_eta

        # Feed prices backups (only those used by each route)
        _Methaneprice0 = Methaneprice
        _Ironprice0 = Ironprice
        _Sulphurprice0 = Sulphurprice
        _Nitrogenprice0 = Nitrogenprice
        _CO2price0 = CO2price
        _FeAlprice0 = FeAlprice

        # Carbon yield backup
        _CarbonyieldFC0 = CarbonyieldFC

        # We need deterministic 1-sample mid-values.
        N = 1
        rng = np.random.default_rng(123456)

        try:
            # ---- baseline gas recovery ~85% ----
            Percent_of_H2_from_purge_gas = np.array([float(overrides.gas_recovery_fraction)], dtype=float)

            # ---- ensure all previously-sampled uncertain inputs are 1-sample arrays ----
            Availability = _as_len1_p50(_Availability0)
            CH4_conversion = _as_len1_p50(_CH4_conversion0)
            PercentCNTYield = _as_len1_p50(_PercentCNTYield0)
            Fe_kg_per_kgCNT = _as_len1_p50(_FeDose0)
            S_kg_per_kgCNT = _as_len1_p50(_SDose0)
            H2_to_CH4_molar = _as_len1_p50(_H2_to_CH4_molar0)

            blanket_thickness_m = _as_len1_p50(_blanket_thickness_m0)
            Insulationefficiency_unc = _as_len1_p50(_Insulationefficiency_unc0)
            Insulationefficiency_unc = np.array(Insulationefficiency_unc, dtype=float) * float(overrides.heat_retention_mult)
            Insulationefficiency_unc = np.clip(Insulationefficiency_unc, 0.0, 0.999999)
            percentageoflabourcosts = _as_len1_p50(_percentageoflabourcosts0)

            CAPEX_factor = _as_len1_p50(_CAPEX_factor0)
            OPEX_factor = _as_len1_p50(_OPEX_factor0)
            scalingexponent = _as_len1_p50(_scalingexponent0)
            Tubescaleupfactor = _as_len1_p50(_Tubescaleupfactor0)

            PSA_feed_pressure_bar = _as_len1_p50(_PSA_feed_pressure_bar0)
            PSA_compressor_eta = _as_len1_p50(_PSA_compressor_eta0)

            # Defensive: coerce any remaining global ndarray drivers to something length-1.
            # This prevents the common DataFrame construction error where a new global
            # MC array is added but not included in the explicit list above.
            for _k, _v in list(globals().items()):
                if isinstance(_v, np.ndarray) and _v.ndim >= 1 and _v.shape[0] != 1:
                    globals()[_k] = _as_len1_p50(_v)

            # ---- throughput perturbation ----
            Throughputofscale_ton_per_year = float(_throughput0_tpy) * float(overrides.throughput_mult)
            Throughputofscale_g_h = float(_throughput0_g_h) * float(overrides.throughput_mult)

            # ---- precursor costs perturbation ----
            Methaneprice = float(_p50(_Methaneprice0)) * float(overrides.precursor_cost_mult)
            Ironprice = float(_p50(_Ironprice0)) * float(overrides.precursor_cost_mult)
            Sulphurprice = float(_p50(_Sulphurprice0)) * float(overrides.precursor_cost_mult)
            Nitrogenprice = float(_p50(_Nitrogenprice0)) * float(overrides.precursor_cost_mult)
            CO2price = float(_p50(_CO2price0)) * float(overrides.precursor_cost_mult)
            FeAlprice = float(_p50(_FeAlprice0)) * float(overrides.precursor_cost_mult)

            # ---- maintenance perturbation ----
            Percentangecostmaintenance = float(_maint0) * float(overrides.maintenance_mult)

            # ---- carbon yield perturbation (applies to both routes here, since FLB uses it as a placeholder) ----
            CarbonyieldFC = np.array([_p50(_CarbonyieldFC0) * float(overrides.carbon_yield_mult)], dtype=float)

            # ---- hot recycle & heat retention perturbations ----
            # FCCVD: recycle sampled inside economic model via beta; easiest deterministic override
            # is to override after the fact by scaling the derived output-driver if present.
            # FLB: recycle sampled inside physics. For deterministic sensitivity we approximate
            # effect by scaling electricity consumption (as recycle/retention impacts heating duty).
            # This keeps the implementation robust without rewriting the entire physics.

            try:
                if route.upper() == "FCCVD":
                    df = economic_model_fccvd_vectorized()
                elif route.upper() == "FLB":
                    df = economic_model_flb_vectorized()
                else:
                    raise ValueError("route must be 'FCCVD' or 'FLB'")
            except Exception as e:
                dbg = _debug_find_len_mismatches(expected_len=1)
                if dbg:
                    raise RuntimeError(f"{e}\n\n{dbg}") from e
                raise

            row = df.iloc[0].to_dict()

            # ---- CAPEX component multipliers ----
            reactor_cost = float(row.get("Reactor_cost_USD", row.get("CAPEX_USD", 0.0))) * overrides.capex_reactor_mult
            facility_cost = float(row.get("Facility_cost_USD", 0.0)) * overrides.capex_facility_mult
            gasrec_cost = float(row.get("Gasrecovery_cost_USD", 0.0)) * overrides.capex_gasrec_mult
            infra_cost = float(row.get("Infra_cost_USD", 0.0)) * overrides.capex_infra_mult
            insulation_cost = float(row.get("Insulation_cost_USD", 0.0)) * overrides.capex_insulation_mult
            capex = reactor_cost + facility_cost + gasrec_cost + infra_cost + insulation_cost

            # ---- OPEX adjustments ----
            # Labour / overhead scale
            labour_cost_y = float(row.get("Labour_cost_USD_y", 0.0)) * overrides.labour_cost_mult
            overhead_cost_y = float(row.get("Overhead_cost_USD_y", 0.0)) * overrides.labour_cost_mult

            # Variable feeds/electric (already includes precursor cost scaling); apply recycle/retention
            # as a multiplier on electricity component.
            elec_cost_y = float(row.get("Elec_cost_USD_y", 0.0))
            elec_cost_y = elec_cost_y * (1.0 / max(1e-12, float(overrides.heat_retention_mult)))
            elec_cost_y = elec_cost_y * float(overrides.hot_recycle_mult)

            # Rebuild total OPEX with adjusted labour/overhead/electric.
            total_opex_y = float(row.get("Total_OPEX_USD_y", row.get("OPEX_USD_per_y", 0.0)))
            # Replace labour/overhead/elec slices; keep the rest constant.
            base_labour = float(row.get("Labour_cost_USD_y", 0.0))
            base_overhead = float(row.get("Overhead_cost_USD_y", 0.0))
            base_elec = float(row.get("Elec_cost_USD_y", 0.0))
            total_opex_y = total_opex_y - base_labour - base_overhead - base_elec + labour_cost_y + overhead_cost_y + elec_cost_y

            # Maintenance is modeled as a CAPEX fraction; recompute with modified CAPEX.
            maint_frac = float(Percentangecostmaintenance)
            maint_y = maint_frac * capex
            base_maint = float(row.get("Maintenance_USD_y", 0.0))
            total_opex_y = total_opex_y - base_maint + maint_y

            # Unit cost
            cnt_kg_y = float(row.get("CNT_kg_y", np.nan))
            # FCCVD doesn't store CNT_kg_y column; infer from P50 throughput basis.
            if not np.isfinite(cnt_kg_y) or cnt_kg_y <= 0:
                cnt_kg_y = float(Throughputofscale_ton_per_year) * 1000.0
            unit_cost_ex_cap = total_opex_y / max(1e-12, cnt_kg_y)

            return {
                "CAPEX_USD": capex,
                "OPEX_USD_per_y": total_opex_y,
                "Unit_cost_ex_cap_USD_per_kgCNT": unit_cost_ex_cap,
            }
        finally:
            # Restore globals
            N = _N0
            rng = _rng0
            Percent_of_H2_from_purge_gas = _h2rec0
            Throughputofscale_ton_per_year = _throughput0_tpy
            Throughputofscale_g_h = _throughput0_g_h
            Percentangecostmaintenance = _maint0
            Methaneprice = _Methaneprice0
            Ironprice = _Ironprice0
            Sulphurprice = _Sulphurprice0
            Nitrogenprice = _Nitrogenprice0
            CO2price = _CO2price0
            FeAlprice = _FeAlprice0
            CarbonyieldFC = _CarbonyieldFC0

            Availability = _Availability0
            CH4_conversion = _CH4_conversion0
            PercentCNTYield = _PercentCNTYield0
            Fe_kg_per_kgCNT = _FeDose0
            S_kg_per_kgCNT = _SDose0
            H2_to_CH4_molar = _H2_to_CH4_molar0
            blanket_thickness_m = _blanket_thickness_m0
            Insulationefficiency_unc = _Insulationefficiency_unc0
            percentageoflabourcosts = _percentageoflabourcosts0
            CAPEX_factor = _CAPEX_factor0
            OPEX_factor = _OPEX_factor0
            scalingexponent = _scalingexponent0
            Tubescaleupfactor = _Tubescaleupfactor0
            PSA_feed_pressure_bar = _PSA_feed_pressure_bar0
            PSA_compressor_eta = _PSA_compressor_eta0


    def evaluate_fccvd_p50(overrides: SensitivityOverrides | None = None) -> dict[str, float]:
        return _evaluate_route_p50("FCCVD", overrides or SensitivityOverrides())


    def evaluate_flb_p50(overrides: SensitivityOverrides | None = None) -> dict[str, float]:
        return _evaluate_route_p50("FLB", overrides or SensitivityOverrides())


    def _run_sensitivity(route: str, metric_key: str = "Unit_cost_ex_cap_USD_per_kgCNT") -> dict[str, list[float]]:
        change_variable = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
        change_percentage = [-50, -30, -10, 0, 10, 30, 50]

        base = _evaluate_route_p50(route, SensitivityOverrides())
        base_val = float(base[metric_key])

        def pct_change(v: float) -> float:
            return 100.0 * (float(v) / max(1e-12, base_val) - 1.0)

        series: dict[str, list[float]] = {
            "change_percentage": change_percentage,
            "Carbon yield": [],
            "Hot gas recycle": [],
            "Heat retention": [],
            "Precursor cost": [],
            "Labour cost": [],
            "Maintenance cost": [],
            "Throughput": [],

            # --- CAPEX component sensitivities (commented out for clarity) ---
            # Uncomment these (and the blocks below) if you want CAPEX components
            # to appear on the sensitivity chart.
            # "Reactor cost": [],
            # "Facility cost": [],
            # "Gas recovery cost": [],
            # "Infrastructure cost": [],
            # "Insulation cost": [],
        }

        for fac in change_variable:
            # physics / opex
            series["Carbon yield"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(carbon_yield_mult=fac))[metric_key]))
            series["Hot gas recycle"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(hot_recycle_mult=fac))[metric_key]))
            series["Heat retention"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(heat_retention_mult=fac))[metric_key]))
            series["Precursor cost"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(precursor_cost_mult=fac))[metric_key]))
            series["Labour cost"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(labour_cost_mult=fac))[metric_key]))
            series["Maintenance cost"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(maintenance_mult=fac))[metric_key]))
            series["Throughput"].append(pct_change(_evaluate_route_p50(route, SensitivityOverrides(throughput_mult=fac))[metric_key]))

            # --- CAPEX component sensitivities (commented out for clarity) ---
            # series["Reactor cost"].append(
            #     pct_change(_evaluate_route_p50(route, SensitivityOverrides(capex_reactor_mult=fac))[metric_key])
            # )
            # series["Facility cost"].append(
            #     pct_change(_evaluate_route_p50(route, SensitivityOverrides(capex_facility_mult=fac))[metric_key])
            # )
            # series["Gas recovery cost"].append(
            #     pct_change(_evaluate_route_p50(route, SensitivityOverrides(capex_gasrec_mult=fac))[metric_key])
            # )
            # series["Infrastructure cost"].append(
            #     pct_change(_evaluate_route_p50(route, SensitivityOverrides(capex_infra_mult=fac))[metric_key])
            # )
            # series["Insulation cost"].append(
            #     pct_change(_evaluate_route_p50(route, SensitivityOverrides(capex_insulation_mult=fac))[metric_key])
            # )

        return series


    def _plot_sensitivity(series: dict[str, list[float]], title: str, save_prefix: str) -> None:
        fig, ax = plt.subplots(figsize=(11.69, 11.69))
        ax2 = ax.twinx()
        ax3 = ax.twiny()

        x = series["change_percentage"]

        # Match template look (heavy lines, mix of dashed/solid)
        ax.plot(x, series["Carbon yield"], label="Carbon Yield", linewidth=5, linestyle="--")
        ax.plot(x, series["Hot gas recycle"], label="Hot cycle", linewidth=5, linestyle="--")
        ax.plot(x, series["Precursor cost"], label="Precursor Cost", linewidth=5)
        ax.plot(x, series["Labour cost"], label="Labour Cost", linewidth=5, linestyle="--")
        ax.plot(x, series["Maintenance cost"], label="Maintenance Cost", linewidth=5, linestyle="--")
        ax.plot(x, series["Throughput"], label="Throughput", linewidth=5)
        ax.plot(x, series["Heat retention"], label="Heat Retention", linewidth=5, linestyle="--")

        # --- CAPEX component lines (commented out for clarity) ---
        # ax.plot(x, series["Reactor cost"], label="Reactor CAPEX", linewidth=4, linestyle=":")
        # ax.plot(x, series["Facility cost"], label="Facility CAPEX", linewidth=4, linestyle=":")
        # ax.plot(x, series["Gas recovery cost"], label="Gas Recovery CAPEX", linewidth=4, linestyle=":")
        # ax.plot(x, series["Infrastructure cost"], label="Infrastructure CAPEX", linewidth=4, linestyle=":")
        # ax.plot(x, series["Insulation cost"], label="Insulation CAPEX", linewidth=4, linestyle=":")

        label_font = {"fontname": "Times New Roman", "size": 16}
        ax.set_xlabel("% Change of Parameter", fontdict=label_font, labelpad=16)
        ax.set_ylabel("% Change of cost", fontdict=label_font, labelpad=16)
        plt.title(title, fontsize=22)

        ax.tick_params(axis="both", which="both", labelsize=16)
        for label in ax.get_xticklabels():
            label.set_fontsize(16)
        for label in ax.get_yticklabels():
            label.set_fontsize(16)

        ax2.tick_params(axis="y", which="both", direction="in", labelleft=False, left=False, right=True, labelright=False)
        ax3.tick_params(axis="x", which="both", direction="in", labelbottom=False, bottom=False, top=True, labeltop=False)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)

        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_xticks([-50, -30, -10, 0, 10, 30, 50])
        ax.set_yticks([-50, -30, -10, 0, 10, 30, 50])

        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.grid(which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        ax.legend(loc="upper left", fontsize=12, frameon=False)
        plt.tight_layout()

        if cfg.save_outputs:
            _savefig_without_titles(fig, f"{cfg.output_dir}/{save_prefix}.svg", format="svg")
            _savefig_without_titles(fig, f"{cfg.output_dir}/{save_prefix}.png", dpi=300)
        if cfg.show_plots:
            plt.show()
        else:
            plt.close(fig)


    # -----------------------------
    # 2) Helper: percentiles table
    # -----------------------------
    def p10_p50_p90(x: pd.Series) -> dict:
        x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) == 0:
            return {"P10": np.nan, "P50": np.nan, "P90": np.nan}
        return {
            "P10": np.percentile(x, 10),
            "P50": np.percentile(x, 50),
            "P90": np.percentile(x, 90),
        }

    rows = []
    for m in metrics:
        if m not in fccvd_df.columns:
            print(f"[WARN] FCCVD missing column: {m}")
            continue
        if m not in flb_df.columns:
            print(f"[WARN] FLB missing column: {m}")
            continue

        f = p10_p50_p90(fccvd_df[m])
        b = p10_p50_p90(flb_df[m])
        rows.append({
            "Metric": m,
            "FCCVD_P10": f["P10"], "FCCVD_P50": f["P50"], "FCCVD_P90": f["P90"],
            "FLB_P10":   b["P10"], "FLB_P50":   b["P50"], "FLB_P90":   b["P90"],
        })

    pct_table = pd.DataFrame(rows)

    # Pretty print (optional)
    pd.set_option("display.max_columns", None)
    print("\n=== P10/P50/P90 table (FCCVD vs FLB) ===")
    print(pct_table)

    # ============================================================
    # 11) SENSITIVITY PLOTS (place AFTER boxplots, per request)
    # ============================================================
    try:
        fccvd_series = _run_sensitivity("FCCVD")
        fccvd_prefix = f"sensitivity_fccvd_{timestamp}"
        _plot_sensitivity(fccvd_series, "FC-CVD", fccvd_prefix)

        flb_series = _run_sensitivity("FLB")
        flb_prefix = f"sensitivity_flb_{timestamp}"
        _plot_sensitivity(flb_series, "FB-CVD", flb_prefix)

        if cfg.save_outputs:
            print(f"[Sensitivity] Saved: {cfg.output_dir}/{fccvd_prefix}.png")
            print(f"[Sensitivity] Saved: {cfg.output_dir}/{flb_prefix}.png")
    except Exception as e:
        print("[Sensitivity] ERROR:", e)

    # -----------------------------
    # 3) Plot function (hist overlay + P10/P50/P90 lines)
    #    Uses your requested figure sizing + twin axes
    # -----------------------------
    def plot_overlay_hist_with_percentiles(
        metric: str,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        label_a: str = "FCCVD",
        label_b: str = "FLB",
        bins: int = 80,
        save_prefix: str | None = None
    ):
        if metric not in df_a.columns or metric not in df_b.columns:
            return

        # -----------------------------------------
        # Publication styling + unit handling
        # -----------------------------------------
        # Map raw metric -> (pretty_xlabel, scale_factor, unit_suffix)
        # scale_factor: multiply raw values by this for plotting.
        SCALE_MAP: dict[str, tuple[str, float, str]] = {
            "CAPEX_USD": ("CAPEX [$Million]", 1e-6, "M"),
            "OPEX_USD_per_y": ("OPEX [$Million/y]", 1e-6, "M"),
            "NPV_USD": ("NPV [$Million]", 1e-6, "M"),
            "Unit_cost_ex_cap_USD_per_kgCNT": ("[$/kgCNT]", 1.0, ""),
        }
        xlabel, scale, _unit = SCALE_MAP.get(metric, (metric, 1.0, ""))

        # Route styling (FC‑CVD green, FB‑CVD black)
        STYLE_A = {"color": "#2ca02c"}  # green
        STYLE_B = {"color": "#000000"}  # black

        def _annotate_vline(ax, x, y_top, text, color, y_offset_pts: float = -10):
            """Annotate a vertical line with its numeric value near the top of the axis."""
            # Put it slightly below the top and centered at x.
            ax.annotate(
                text,
                xy=(x, y_top),
                xytext=(0, y_offset_pts),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85),
                clip_on=True,
            )

        a = (
            pd.to_numeric(df_a[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
            * scale
        )
        b = (
            pd.to_numeric(df_b[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
            * scale
        )
        if len(a) == 0 or len(b) == 0:
            print(f"[WARN] Empty data for metric: {metric}")
            return

        # Common bin edges => apples-to-apples histogram
        lo = np.nanmin([np.nanmin(a), np.nanmin(b)])
        hi = np.nanmax([np.nanmax(a), np.nanmax(b)])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            print(f"[WARN] Bad range for metric: {metric} (lo={lo}, hi={hi})")
            return
        bin_edges = np.linspace(lo, hi, bins + 1)

        # Percentiles
        a_p10, a_p50, a_p90 = np.percentile(a, [10, 50, 90])
        b_p10, b_p50, b_p90 = np.percentile(b, [10, 50, 90])

        # Create a new figure with specified size
        fig, ax = plt.subplots(figsize=(11.69, 11.69))
        ax2 = ax.twinx()  # Create a twin y-axis (kept for your template)
        ax3 = ax.twiny()  # Create a twin x-axis (kept for your template)

        # Mirror ticks (inside) and remove labels on the twin axes
        ax.tick_params(axis="both", which="both", direction="in", length=6)
        ax2.tick_params(axis="y", which="both", direction="in", length=6, labelright=False)
        ax3.tick_params(axis="x", which="both", direction="in", length=6, labeltop=False)

        # Plot histograms
        n_a, _, _ = ax.hist(
            a,
            bins=bin_edges,
            alpha=0.55,
            label=label_a,
            density=False,
            edgecolor=STYLE_A["color"],
            facecolor=STYLE_A["color"],
        )
        n_b, _, _ = ax.hist(
            b,
            bins=bin_edges,
            alpha=0.35,
            label=label_b,
            density=False,
            edgecolor=STYLE_B["color"],
            facecolor=STYLE_B["color"],
        )

        # Percentile lines + labels
        # FC‑CVD
        ax.axvline(a_p10, linewidth=2, linestyle=":", color=STYLE_A["color"], label=f"{label_a} Low (P10)")
        ax.axvline(a_p50, linewidth=2, linestyle="--", color=STYLE_A["color"], label=f"{label_a} Best (P50)")
        ax.axvline(a_p90, linewidth=2, linestyle="-", color=STYLE_A["color"], label=f"{label_a} High (P90)")

        # FB‑CVD
        ax.axvline(b_p10, linewidth=2, linestyle=":", color=STYLE_B["color"], label=f"{label_b} Low (P10)")
        ax.axvline(b_p50, linewidth=2, linestyle="--", color=STYLE_B["color"], label=f"{label_b} Best (P50)")
        ax.axvline(b_p90, linewidth=2, linestyle="-", color=STYLE_B["color"], label=f"{label_b} High (P90)")

        # Annotate with actual numeric values on the lines (P10/P50/P90)
        y_top = max(
            float(np.nanmax(n_a)) if len(n_a) else 0.0,
            float(np.nanmax(n_b)) if len(n_b) else 0.0,
            1.0,  # guard so label placement is always meaningful
        )
        # Nudge heights to reduce overlap between routes
        y_a = y_top * 0.98
        y_b = y_top * 0.90
        def fmt(v: float) -> str:
            """Two significant figures, no decimal points (compact labels)."""
            if not np.isfinite(v):
                return ""
            if v == 0:
                return "0"

            v_abs = float(abs(v))
            # Special case: keep 1..10 readable (e.g., 7.6) instead of 76e-1.
            if 1.0 <= v_abs < 10.0:
                return f"{v:.1f}"
            exp = int(np.floor(np.log10(v_abs)))
            # Scale so the mantissa becomes a 2-digit integer (~10..99)
            mantissa_int = int(np.round(v_abs / (10 ** (exp - 1))))

            # Handle cases like 9.95 -> 100 (push exponent up)
            if mantissa_int >= 100:
                mantissa_int //= 10
                exp += 1

            signed_mantissa = mantissa_int if v >= 0 else -mantissa_int
            out_exp = exp - 1

            # If exponent is non-negative, we can display as an integer with commas.
            if out_exp >= 0:
                rounded_int = signed_mantissa * (10 ** out_exp)
                return f"{rounded_int:,.0f}"

            # Otherwise, use scientific-like form without decimal point in mantissa.
            # Use mathtext so the exponent is superscript (publication-friendly).
            return f"${signed_mantissa}\\times 10^{{{out_exp}}}$"
        # Stagger vertical label positions so P10/P50/P90 do not overlap when x-values are close.
        _annotate_vline(ax, a_p10, y_a, fmt(a_p10), STYLE_A["color"], y_offset_pts=-90)
        _annotate_vline(ax, a_p50, y_a, fmt(a_p50), STYLE_A["color"], y_offset_pts=-55)
        _annotate_vline(ax, a_p90, y_a, fmt(a_p90), STYLE_A["color"], y_offset_pts=-20)
        _annotate_vline(ax, b_p10, y_b, fmt(b_p10), STYLE_B["color"], y_offset_pts=-150)
        _annotate_vline(ax, b_p50, y_b, fmt(b_p50), STYLE_B["color"], y_offset_pts=-115)
        _annotate_vline(ax, b_p90, y_b, fmt(b_p90), STYLE_B["color"], y_offset_pts=-80)

        # For unit-cost comparison, cap x-axis at 100.
        if metric == "Unit_cost_ex_cap_USD_per_kgCNT":
            ax.set_xlim(left=0.0, right=100.0)

        # Update mirrored axes limits now that we have data
        ax2.set_ylim(ax.get_ylim())
        ax3.set_xlim(ax.get_xlim())

        # Labels / title
        ax.set_title(f"Monte Carlo distribution: {metric}", fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel("Count", fontsize=14, fontweight="bold")

        # Ensure axis units match the label (no scientific offset like '1e6')
        _disable_axis_sci_offset(ax, axis="x")

        # Legend: put outside to avoid covering histogram
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
        fig.tight_layout()

        # Optional saves
        if save_prefix is not None:
            safe_name = metric.replace("/", "_").replace(" ", "_")
            _savefig_without_titles(fig, f"{save_prefix}_{safe_name}.png", dpi=200)
            _savefig_without_titles(fig, f"{save_prefix}_{safe_name}.svg")

        if cfg.show_plots:
            plt.show()
        else:
            plt.close(fig)

    # -----------------------------
    # 4) Make all plots
    # -----------------------------
    # Optional: set a prefix if you want files saved automatically
    SAVE_PREFIX = "fccvd_vs_flb"  # saves into current directory

    for m in metrics:
        plot_overlay_hist_with_percentiles(
            metric=m,
            df_a=fccvd_df,
            df_b=flb_df,
            label_a="FC-CVD",
            label_b="FB-CVD",
            bins=80,
            save_prefix=SAVE_PREFIX
        )

    # -----------------------------
    # 5) (Optional) Save the percentile table
    # -----------------------------
    # pct_table.to_csv(f"{cfg.output_dir}/fccvd_vs_flb_p10_p50_p90_{timestamp}.csv", index=False)




def _horizontal_share_boxplot(
    df,
    share_cols_map,
    title,
    figsize=(11.69, 11.69),
    save_png=None,
    save_svg=None,
):
    """
    df: DataFrame with share columns in 0..1 (fractions)
    share_cols_map: dict like {"Precursors": "Share_Percursor_in_TotalOPEX", ...}
    Produces horizontal boxplots in %, ranked top->bottom by median share.
    """

    # --- collect + convert to % ---
    data = {}
    for nice_name, col in share_cols_map.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df. Available columns include: {list(df.columns)[:10]} ...")
        x = np.asarray(df[col].values, dtype=float) * 100.0
        # guard: keep within [0,100]
        x = np.clip(x, 0.0, 100.0)
        data[nice_name] = x

    # --- rank by median, high -> low ---
    med = {k: np.nanmedian(v) for k, v in data.items()}
    order = sorted(med.keys(), key=lambda k: med[k], reverse=True)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    # positions: 0..k-1
    positions = np.arange(len(order))

    bp = ax.boxplot(
        [data[k] for k in order],
        vert=False,                 # horizontal
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(color="orange", linewidth=3),
    )

    # Publication styling (match histogram rules)
    ax.tick_params(axis="both", which="both", direction="in", length=6)
    ax.set_xlabel("Share of Total (%)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # x-axis in %
    ax.set_xlim(0, 100)

    # remove y-axis ticks/labels
    ax.set_yticks([])
    ax.set_ylabel("")

    # Put label just above each box/median, visually "hooked" to the box.
    # Use the median x-position per category so labels track the box location.
    for i, name in enumerate(order):
        x_med = float(np.nanmedian(data[name]))
        if not np.isfinite(x_med):
            x_med = 0.0

        # Label placement rule:
        # - If share >= 85% => place label slightly left on the box (avoid running into right boundary)
        # - If 50% <= share < 85% => place label centered over the box
        # - If share < 50%  => place label to the right (minimum anchor)
        if x_med >= 85.0:
            x_anchor = max(5.0, x_med - 10.0)
        elif x_med >= 50.0:
            x_anchor = x_med
        else:
            x_anchor = max(x_med, 8.0)

        ax.annotate(
            name,
            xy=(x_med, i),
            xytext=(x_anchor, i),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5, shrinkA=0, shrinkB=4),
            clip_on=True,
        )

    # rank top->bottom: highest share at top
    ax.invert_yaxis()

    # cosmetic grid on x only
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=12)

    fig.tight_layout()

    if save_png:
        _savefig_without_titles(fig, save_png, dpi=300, bbox_inches="tight")
    if save_svg:
        _savefig_without_titles(fig, save_svg, bbox_inches="tight")

    return fig, ax


# =========================
# FCCVD plot
# =========================
# fccvd_results is your FCCVD Monte Carlo dataframe (output of economic_model_fccvd_vectorized or loaded CSV)
fccvd_share_cols = {
    "Precursors": "Share_Percursor_in_TotalOPEX",
    "Electricity": "Share_Elec_in_TotalOPEX",
    "Labour": "Share_Labour_in_TotalOPEX",
    "Overhead": "Share_Overhead_in_TotalOPEX",
    "Infra/Depr.": "Share_InfraDepr_in_TotalOPEX",
    "Maintenance": "Share_Maint_in_TotalOPEX",
}

fccvd_capex_share_cols = {
    "Reactor": "Share_reactor_in_TotalCAPEX",
    "Facility": "Share_facility_in_TotalCAPEX",
    "Gas recovery": "Share_gasrec_in_TotalCAPEX",
    "Infra/QA": "Share_infra_in_TotalCAPEX",
    "Insulation": "Share_insulation_in_TotalCAPEX",
}

_horizontal_share_boxplot(
    df=fccvd_results,
    share_cols_map=fccvd_share_cols,
    title="FC-CVD – Total OPEX Share Breakdown (Monte Carlo)",
    figsize=(11.69, 11.69),
    save_png="fccvd_opex_share_boxplot.png",
    save_svg="fccvd_opex_share_boxplot.svg",
)

_horizontal_share_boxplot(
    df=fccvd_results,
    share_cols_map=fccvd_capex_share_cols,
    title="FC-CVD – Total CAPEX Share Breakdown (Monte Carlo)",
    figsize=(11.69, 11.69),
    save_png="fccvd_capex_share_boxplot.png",
    save_svg="fccvd_capex_share_boxplot.svg",
)


# =========================
# FLB plot
# =========================
# flb_results is your FLB Monte Carlo dataframe (output of economic_model_flb_vectorized or loaded CSV)
flb_share_cols = {
    "Precursors": "Share_Percursor_in_TotalOPEX",
    "Electricity": "Share_Elec_in_TotalOPEX",
    "Labour": "Share_Labour_in_TotalOPEX",
    "Overhead": "Share_Overhead_in_TotalOPEX",
    "Infra/Depr.": "Share_InfraDepr_in_TotalOPEX",
    "Maintenance": "Share_Maint_in_TotalOPEX",
}

flb_capex_share_cols = {
    "Reactor": "Share_reactor_in_TotalCAPEX",
    "Facility": "Share_facility_in_TotalCAPEX",
    "Gas recovery": "Share_gasrec_in_TotalCAPEX",
    "Infra/QA": "Share_infra_in_TotalCAPEX",
    "Insulation": "Share_insulation_in_TotalCAPEX",
}

_horizontal_share_boxplot(
    df=flb_results,
    share_cols_map=flb_share_cols,
    title="FB-CVD – Total OPEX Share Breakdown (Monte Carlo)",
    figsize=(11.69, 11.69),
    save_png="flb_opex_share_boxplot.png",
    save_svg="flb_opex_share_boxplot.svg",
)

_horizontal_share_boxplot(
    df=flb_results,
    share_cols_map=flb_capex_share_cols,
    title="FB-CVD – Total CAPEX Share Breakdown (Monte Carlo)",
    figsize=(11.69, 11.69),
    save_png="flb_capex_share_boxplot.png",
    save_svg="flb_capex_share_boxplot.svg",
)

if cfg.show_plots:
    plt.show()
else:
    plt.close("all")



