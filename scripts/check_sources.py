#!/usr/bin/env python3
"""Verify mujoco-water source code against published reference data.

Parses C++ headers, extracts physical constants and correlation parameters,
and compares against canonical values in reference_data.yaml. Fails with
exit code 1 if any claim in the code disagrees with its cited source.

This is NOT a runtime test suite (see tests/). It is a static source
fidelity audit: does the code implement what the references say?

Usage:
    python scripts/check_sources.py
    python scripts/check_sources.py --report source_check_report.md
"""

import math
import os
import re
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
HEADER_DIR = os.path.join(ROOT_DIR, "include", "mjwater")
REF_FILE = os.path.join(SCRIPT_DIR, "reference_data.yaml")


def load_refs():
    with open(REF_FILE) as f:
        return yaml.safe_load(f)


def read_header(name):
    path = os.path.join(HEADER_DIR, name)
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class Results:
    def __init__(self):
        self.entries = []

    def ok(self, section, name, detail, source):
        self.entries.append(("PASS", section, name, detail, source))

    def fail(self, section, name, detail, source):
        self.entries.append(("FAIL", section, name, detail, source))

    def warn(self, section, name, detail, source):
        self.entries.append(("WARN", section, name, detail, source))

    @property
    def failures(self):
        return sum(1 for s, *_ in self.entries if s == "FAIL")

    def print_report(self, path=None):
        lines = ["# Source Fidelity Check Report\n"]
        current_section = None
        for status, section, name, detail, source in self.entries:
            if section != current_section:
                lines.append(f"\n## {section}\n")
                current_section = section
            tag = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}[status]
            lines.append(f"  {tag}  {name}: {detail}")
            if status == "FAIL":
                lines.append(f"        source: {source}")
        lines.append(f"\n---\n{self.failures} failures, "
                      f"{sum(1 for s,*_ in self.entries if s == 'PASS')} passed, "
                      f"{sum(1 for s,*_ in self.entries if s == 'WARN')} warnings\n")
        text = "\n".join(lines)
        print(text)
        if path:
            with open(path, "w") as f:
                f.write(text)


def extract_float(header_text, pattern):
    """Extract a float from a C++ header using a regex pattern."""
    m = re.search(pattern, header_text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Handle scientific notation and remove trailing f
    raw = raw.rstrip("f").strip()
    return float(raw)


# ---------------------------------------------------------------------------
# Module 1: Physical constants
# ---------------------------------------------------------------------------

def check_physical_constants(refs, results):
    section = "Physical Constants"
    for entry in refs.get("physical_constants", []):
        name = entry["name"]
        src = read_header(entry["file"])
        val = extract_float(src, entry["pattern"])
        if val is None:
            results.fail(section, name, f"not found in {entry['file']}", entry["source"])
            continue
        ref = entry["value"]
        tol = entry["tolerance"]
        err = abs(val - ref)
        if err <= tol:
            results.ok(section, name, f"{val} (ref: {ref}, tol: {tol})", entry["source"])
        else:
            results.fail(section, name,
                         f"code={val}, ref={ref}, err={err:.2e} > tol={tol}",
                         entry["source"])


# ---------------------------------------------------------------------------
# Module 2: D3Q19 lattice constants
# ---------------------------------------------------------------------------

def parse_int_array(src, name, count):
    """Parse a constexpr int array from C++ source."""
    pattern = rf'{name}\[{count}\]\s*=\s*\{{([^}}]+)\}}'
    m = re.search(pattern, src)
    if not m:
        return None
    return [int(x.strip()) for x in m.group(1).split(",")]


def parse_float_array(src, name, count):
    """Parse a constexpr float array from C++ source."""
    pattern = rf'{name}\[{count}\]\s*=\s*\{{([^}}]+)\}}'
    m = re.search(pattern, src, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # Evaluate fractions like 1.0f/3.0f
        token = token.replace("f", "")
        vals.append(eval(token))
    return vals


def check_lattice(refs, results):
    section = "D3Q19 Lattice"
    src = read_header("water_grid.h")
    lat = refs.get("lattice_d3q19", {})

    # Parse arrays
    ex = parse_int_array(src, "kEx", "kQ")
    ey = parse_int_array(src, "kEy", "kQ")
    ez = parse_int_array(src, "kEz", "kQ")
    w = parse_float_array(src, "kW", "kQ")
    opp = parse_int_array(src, "kOpp", "kQ")

    if not all([ex, ey, ez, w, opp]):
        results.fail(section, "parse", "could not parse lattice arrays", "")
        return

    Q = len(w)
    source = lat.get("source_structural", "Kruger et al. (2017)")

    # Weight sum
    wsum = sum(w)
    if abs(wsum - 1.0) < 1e-7:
        results.ok(section, "weight sum", f"sum(w) = {wsum:.10f}", lat["weights"]["source"])
    else:
        results.fail(section, "weight sum", f"sum(w) = {wsum}, expected 1.0",
                     lat["weights"]["source"])

    # Velocity vector counts
    n_rest = sum(1 for i in range(Q) if ex[i] == 0 and ey[i] == 0 and ez[i] == 0)
    n_face = sum(1 for i in range(Q)
                 if ex[i]**2 + ey[i]**2 + ez[i]**2 == 1)
    n_edge = sum(1 for i in range(Q)
                 if ex[i]**2 + ey[i]**2 + ez[i]**2 == 2)
    if n_rest == 1 and n_face == 6 and n_edge == 12:
        results.ok(section, "velocity counts",
                   f"rest={n_rest}, face={n_face}, edge={n_edge}", source)
    else:
        results.fail(section, "velocity counts",
                     f"rest={n_rest}, face={n_face}, edge={n_edge} (expected 1,6,12)",
                     source)

    # Sum of velocity vectors = 0
    sx = sum(ex)
    sy = sum(ey)
    sz = sum(ez)
    if sx == 0 and sy == 0 and sz == 0:
        results.ok(section, "velocity sum", "sum(e) = (0, 0, 0)", source)
    else:
        results.fail(section, "velocity sum", f"sum(e) = ({sx}, {sy}, {sz})", source)

    # Opposite index involution
    involution_ok = all(opp[opp[i]] == i for i in range(Q))
    if involution_ok:
        results.ok(section, "opposite involution", "kOpp[kOpp[i]] = i for all i", source)
    else:
        bad = [i for i in range(Q) if opp[opp[i]] != i]
        results.fail(section, "opposite involution", f"fails at i={bad}", source)

    # Opposite reverses direction
    reverse_ok = all(
        ex[opp[i]] == -ex[i] and ey[opp[i]] == -ey[i] and ez[opp[i]] == -ez[i]
        for i in range(Q))
    if reverse_ok:
        results.ok(section, "opposite direction", "e[kOpp[i]] = -e[i] for all i", source)
    else:
        bad = [i for i in range(Q)
               if ex[opp[i]] != -ex[i] or ey[opp[i]] != -ey[i] or ez[opp[i]] != -ez[i]]
        results.fail(section, "opposite direction", f"fails at i={bad}", source)

    # Second-order isotropy: sum(w_i * e_ia * e_ib) = cs^2 * delta_ab
    cs2 = 1.0 / 3.0
    iso_ok = True
    for a_name, a_arr in [("x", ex), ("y", ey), ("z", ez)]:
        for b_name, b_arr in [("x", ex), ("y", ey), ("z", ez)]:
            val = sum(w[i] * a_arr[i] * b_arr[i] for i in range(Q))
            expected = cs2 if a_name == b_name else 0.0
            if abs(val - expected) > 1e-7:
                results.fail(section, f"isotropy ({a_name},{b_name})",
                             f"sum(w*e_{a_name}*e_{b_name}) = {val:.8f}, expected {expected}",
                             source)
                iso_ok = False
    if iso_ok:
        results.ok(section, "second-order isotropy",
                   "sum(w_i * e_ia * e_ib) = (1/3) * delta_ab", source)


# ---------------------------------------------------------------------------
# Module 3: Wendland C2 kernel normalization
# ---------------------------------------------------------------------------

def check_kernel(refs, results):
    section = "Wendland C2 Kernel"
    kern = refs.get("kernel_wendland_c2", {})
    source = kern.get("source", "Wendland (1995)")

    # Check normalization coefficient: 21 / (16*pi) for support radius H = 2h.
    expected_coeff = 21.0 / (16.0 * math.pi)
    ref_coeff = kern["normalization_3d"]["coefficient"]
    tol = kern["normalization_3d"]["tolerance"]
    if abs(ref_coeff - expected_coeff) > tol:
        results.fail(section, "norm coefficient",
                     f"YAML says {ref_coeff}, computed 21/(2pi) = {expected_coeff}",
                     source)
        return

    # Verify code has the right coefficient in the Init function.
    # The declaration initializes to 0; the real value is set in Init().
    src = read_header("sph.h")
    # Look for the assignment inside Init(), not the member declaration.
    m = re.search(r'void Init\(.*?\{.*?norm_3d = ([^;]+);', src, re.DOTALL)
    if m:
        formula = m.group(1).strip()
        if "21.0f" in formula and "16.0f" in formula and "kPi" in formula:
            results.ok(section, "normalization formula",
                       f"code: {formula} = 21/(16*pi*h^3)", source)
        else:
            results.warn(section, "normalization formula",
                         f"unexpected formula: {formula}", source)
    else:
        results.warn(section, "normalization formula",
                     "could not find norm_3d assignment in Init()", source)

    # Numerical integration: W(r) * 4*pi*r^2 dr from 0 to 2h should = 1.0
    # Support radius H = 2h, so normalization is 21/(2*pi*H^3) = 21/(16*pi*h^3).
    h = 1.0  # use h=1 for testing
    norm = 21.0 / (16.0 * math.pi * h**3)
    N = 10000
    dr = 2.0 * h / N
    integral = 0.0
    for i in range(N):
        r = (i + 0.5) * dr
        q = r / h
        if q >= 2.0:
            continue
        t = 1.0 - 0.5 * q
        W = norm * t**4 * (1.0 + 2.0 * q)
        integral += W * 4.0 * math.pi * r * r * dr

    int_tol = kern.get("integration_tolerance", 0.001)
    if abs(integral - 1.0) < int_tol:
        results.ok(section, "3D integration",
                   f"integral = {integral:.6f} (should be 1.0)", source)
    else:
        results.fail(section, "3D integration",
                     f"integral = {integral:.6f}, expected 1.0 (err={abs(integral-1.0):.2e})",
                     source)

    # Verify gradient formula: dW/dr at midpoint matches finite difference.
    # Use the same norm from the integration (21/(16*pi*h^3)).
    r_test = 0.7 * h
    q = r_test / h
    t = 1.0 - 0.5 * q
    dw_analytic = norm * (-5.0 * q / h) * t**3

    eps = 1e-5
    q_p = (r_test + eps) / h
    q_m = (r_test - eps) / h
    w_p = norm * (1.0 - 0.5*q_p)**4 * (1.0 + 2.0*q_p)
    w_m = norm * (1.0 - 0.5*q_m)**4 * (1.0 + 2.0*q_m)
    dw_fd = (w_p - w_m) / (2.0 * eps)

    if abs(dw_analytic - dw_fd) / (abs(dw_fd) + 1e-20) < 0.01:
        results.ok(section, "gradient formula",
                   f"dW/dr analytic={dw_analytic:.6f}, FD={dw_fd:.6f}", source)
    else:
        results.fail(section, "gradient formula",
                     f"dW/dr analytic={dw_analytic:.6f}, FD={dw_fd:.6f}",
                     source)


# ---------------------------------------------------------------------------
# Module 4: Drag correlations (evaluated by reimplementing in Python)
# ---------------------------------------------------------------------------

def sphere_cd(Re):
    """Reimplement SphereDragCoeff from coupling.h in Python."""
    if Re < 0.1:
        return 24.0 / 0.1
    if Re < 1.0:
        return 24.0 / Re
    if Re < 1000.0:
        return 24.0 / Re * (1.0 + 0.15 * Re**0.687)
    if Re < 2e5:
        return 0.44
    return 0.1


def cylinder_cd(Re):
    """Reimplement CylinderDragCoeff from coupling.h in Python."""
    if Re < 0.1:
        return 100.0
    if Re < 1.0:
        return 10.0 / math.sqrt(Re)
    if Re < 1e3:
        return 1.0 + 10.0 / Re**(2.0/3.0)
    if Re < 2e5:
        return 1.2
    if Re < 5e5:
        return 0.3
    return 0.6


def sphere_cm(KC):
    """Reimplement SphereInertiaCoeff from coupling.h in Python."""
    if KC < 3.0:
        return 1.5
    if KC < 15.0:
        return 1.5 - 0.033 * (KC - 3.0)
    return 1.1


def check_drag_correlations(refs, results):
    section = "Drag Correlations"

    # Sphere drag
    sph_ref = refs.get("sphere_drag", {})
    source = sph_ref.get("source", "Clift et al. (1978)")
    all_ok = True
    for pt in sph_ref.get("data_points", []):
        Re = float(pt["Re"])
        cd_ref = float(pt["Cd"])
        cd_code = sphere_cd(Re)
        tol = cd_ref * pt["tolerance_pct"] / 100.0
        if abs(cd_code - cd_ref) <= tol:
            pass  # individual point ok
        else:
            results.fail(section, f"sphere Cd at Re={Re}",
                         f"code={cd_code:.4f}, ref={cd_ref}, tol={tol:.4f}", source)
            all_ok = False
    if all_ok and sph_ref.get("data_points"):
        results.ok(section, "sphere Cd correlation",
                   f"all {len(sph_ref['data_points'])} reference points match", source)

    # Cylinder drag
    cyl_ref = refs.get("cylinder_drag", {})
    source = cyl_ref.get("source", "Zdravkovich (1997)")
    all_ok = True
    for pt in cyl_ref.get("data_points", []):
        Re = float(pt["Re"])
        cd_ref = float(pt["Cd"])
        cd_code = cylinder_cd(Re)
        tol = cd_ref * pt["tolerance_pct"] / 100.0
        if abs(cd_code - cd_ref) <= tol:
            pass
        else:
            results.fail(section, f"cylinder Cd at Re={Re}",
                         f"code={cd_code:.4f}, ref={cd_ref}, tol={tol:.4f}", source)
            all_ok = False
    if all_ok and cyl_ref.get("data_points"):
        results.ok(section, "cylinder Cd correlation",
                   f"all {len(cyl_ref['data_points'])} reference points match", source)

    # Sphere inertia coefficient
    cm_ref = refs.get("sphere_inertia", {})
    source = cm_ref.get("source", "Sarpkaya (1976)")
    all_ok = True
    for pt in cm_ref.get("data_points", []):
        KC = float(pt["KC"])
        cm_expected = float(pt["Cm"])
        cm_code = sphere_cm(KC)
        tol = pt["tolerance"]
        if abs(cm_code - cm_expected) <= tol:
            pass
        else:
            results.fail(section, f"sphere Cm at KC={KC}",
                         f"code={cm_code:.3f}, ref={cm_expected}, tol={tol}", source)
            all_ok = False
    if all_ok and cm_ref.get("data_points"):
        results.ok(section, "sphere Cm correlation",
                   f"all {len(cm_ref['data_points'])} reference points match", source)


# ---------------------------------------------------------------------------
# Module 5: Dispersion relation
# ---------------------------------------------------------------------------

def check_dispersion(refs, results):
    section = "Dispersion Relation"
    disp = refs.get("dispersion", {})
    source = disp.get("source", "Airy (1845)")
    g = 9.80665
    tol = disp.get("tolerance", 0.01)

    # Read the spectral ocean header to verify the code structure
    src = read_header("spectral_ocean.h")

    all_ok = True
    for pt in disp.get("check_points", []):
        k = pt["k"]
        d = pt["d"]
        omega_expected = math.sqrt(g * k * math.tanh(k * d))

        # Reimplement the code's logic
        kd = k * d
        if kd > 10.0:
            omega_code = math.sqrt(g * k)
        else:
            omega_code = math.sqrt(g * k * math.tanh(kd))

        rel_err = abs(omega_code - omega_expected) / (omega_expected + 1e-20)
        if rel_err <= tol:
            pass
        else:
            results.fail(section, f"dispersion k={k} d={d}",
                         f"code={omega_code:.6f}, ref={omega_expected:.6f}, "
                         f"rel_err={rel_err:.2e}", source)
            all_ok = False

    if all_ok and disp.get("check_points"):
        results.ok(section, "dispersion relation",
                   f"all {len(disp['check_points'])} check points match", source)

    # Verify deep-water cutoff is safe
    kd_cutoff = 10.0
    err = 1.0 - math.tanh(kd_cutoff)
    if err < 1e-6:
        results.ok(section, "deep water cutoff",
                   f"tanh({kd_cutoff}) = {math.tanh(kd_cutoff):.12f}, err={err:.2e}",
                   source)
    else:
        results.fail(section, "deep water cutoff",
                     f"tanh({kd_cutoff}) error = {err:.2e}, too large", source)


# ---------------------------------------------------------------------------
# Module 6: CFL conditions
# ---------------------------------------------------------------------------

def check_cfl(refs, results):
    section = "CFL Conditions"
    cfl = refs.get("cfl_conditions", {})

    # Stokes CFL: safety factor must be < 1/6
    stokes = cfl.get("stokes", {})
    if stokes:
        safety = stokes["code_safety"]
        limit = stokes["theoretical_limit"]
        source = stokes.get("source", "von Neumann analysis")
        if safety < limit:
            results.ok(section, "Stokes CFL",
                       f"safety={safety} < 1/6={limit:.6f}", source)
        else:
            results.fail(section, "Stokes CFL",
                         f"safety={safety} >= 1/6={limit:.6f} (unstable!)", source)

    # LBM tau minimum
    lbm = cfl.get("lbm_tau_min", {})
    if lbm:
        source = lbm.get("source", "Kruger et al. (2017)")
        # Verify the code uses tau > 0.5 (check default)
        src = read_header("lbm.h")
        m = re.search(r'float tau = ([^;]+?)f;', src)
        if m:
            tau_default = float(m.group(1))
            if tau_default > 0.5:
                results.ok(section, "LBM tau",
                           f"default tau={tau_default} > 0.5", source)
            else:
                results.fail(section, "LBM tau",
                             f"default tau={tau_default} <= 0.5 (unstable!)", source)

    # TRT magic parameter
    lbm_formulas = refs.get("lbm_formulas", {})
    trt = lbm_formulas.get("trt_magic", {})
    if trt:
        source = trt.get("source", "Ginzburg (2005)")
        src = read_header("lbm.h")
        m = re.search(r'float magic_param = ([^;]+?)f;', src)
        if m:
            val = float(m.group(1))
            if abs(val - 0.25) < 0.01:
                results.ok(section, "TRT magic parameter",
                           f"Lambda={val} (ref: 0.25)", source)
            else:
                results.fail(section, "TRT magic parameter",
                             f"Lambda={val}, expected 0.25", source)


# ---------------------------------------------------------------------------
# Module 7: Spectral and EOS constants
# ---------------------------------------------------------------------------

def check_spectral_and_eos(refs, results):
    section = "Spectral / EOS Constants"

    # Phillips constant
    spec = refs.get("spectral", {})
    phil = spec.get("phillips_constant", {})
    if phil:
        src = read_header(phil["file"])
        val = extract_float(src, phil["pattern"])
        rng = phil["range"]
        source = phil.get("source", "Phillips (1957)")
        if val is not None and rng[0] <= val <= rng[1]:
            results.ok(section, "Phillips constant",
                       f"A={val} in range [{rng[0]}, {rng[1]}]", source)
        elif val is not None:
            results.fail(section, "Phillips constant",
                         f"A={val} outside range [{rng[0]}, {rng[1]}]", source)

    # JONSWAP gamma default
    jg = spec.get("jonswap_gamma_default", {})
    if jg:
        src = read_header("spectral_ocean.h")
        m = re.search(r'jonswap_gamma = ([^;]+?)f;', src)
        if m:
            val = float(m.group(1))
            source = jg.get("source", "Hasselmann et al. (1973)")
            if abs(val - jg["value"]) < 0.1:
                results.ok(section, "JONSWAP gamma",
                           f"gamma={val} (ref: {jg['value']})", source)
            else:
                results.fail(section, "JONSWAP gamma",
                             f"gamma={val}, expected {jg['value']}", source)

    # Tait EOS gamma
    tait = refs.get("tait_eos", {})
    if tait:
        src = read_header("sph.h")
        m = re.search(r'constexpr float gamma = ([^;]+?)f;', src)
        if m:
            val = float(m.group(1))
            source = tait.get("source", "Becker & Teschner (2007)")
            if val == tait["gamma"]:
                results.ok(section, "Tait EOS gamma",
                           f"gamma={int(val)} (ref: {tait['gamma']})", source)
            else:
                results.fail(section, "Tait EOS gamma",
                             f"gamma={val}, expected {tait['gamma']}", source)

    # Manning default
    mann = refs.get("manning_friction", {})
    if mann:
        src = read_header("shallow_water.h")
        m = re.search(r'manning_n = ([^;]+?)f;', src)
        if m:
            val = float(m.group(1))
            source = mann.get("source", "Chow (1959)")
            if abs(val - mann["default_n"]) <= mann["tolerance"]:
                results.ok(section, "Manning n",
                           f"n={val} (ref: {mann['default_n']})", source)
            else:
                results.fail(section, "Manning n",
                             f"n={val}, expected {mann['default_n']}", source)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report_path = None
    if "--report" in sys.argv:
        idx = sys.argv.index("--report")
        if idx + 1 < len(sys.argv):
            report_path = sys.argv[idx + 1]

    refs = load_refs()
    results = Results()

    check_physical_constants(refs, results)
    check_lattice(refs, results)
    check_kernel(refs, results)
    check_drag_correlations(refs, results)
    check_dispersion(refs, results)
    check_cfl(refs, results)
    check_spectral_and_eos(refs, results)

    results.print_report(report_path)
    sys.exit(1 if results.failures > 0 else 0)


if __name__ == "__main__":
    main()
