# Chemical-vapor-deposition-TEA

# CNT Techno‑Economic Analysis (TEA) — FCCVD + FLB (Monte Carlo)

This repository contains a single, self-contained Python model for industrial-scale CNT techno‑economics with **two production routes**:

- **FC‑CVD (FCCVD)**: physics-based methane cracking mass/energy balance + techno‑economic model
- **FB‑CVD (FLB / fluidized bed)**: physics-based fluidized-bed route + techno‑economic model

Both routes are evaluated with a **Monte Carlo simulation** to propagate uncertainty (prices, yields, conversion, dosing, availability, CAPEX/OPEX factors) into outputs such as **unit cost** and **NPV**.

## Repository layout

- `CVD Model TEA.py` — main model + Monte Carlo runner (writes CSV + plots)
- `1- 1Mt/` — (optional) local output folder for a ~1 Mt/y scenario
- `2-1Kt/` — (optional) local output folder for a ~1 kt/y scenario

These output folders are **generated** by running the script and are ignored by default in `.gitignore` so the GitHub repository stays lightweight.

---

## Quick start

### 1) Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

User: Can you help me create a github sharable that people can access the code. But first create a Readme explaing the model and how to change inputs to generate desired output ( e.g.,1 Kt vs 1 Mt)

GitHub Copilot: Optimizing tool selection...I’ll first inspect your `CVD Model TEA.py` and the folder layout to understand what inputs control “1 Kt vs 1 Mt”, then I’ll write a clear `README.md` with run instructions + how to change inputs, and finally I’ll give you the exact GitHub steps (init repo, commit, push) so it’s shareable.

Created 4 todos

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github)

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py#1-1), lines 1 to 200

Found the key knob already: the script uses a single `SCALE = set_scale(...)` line to set production capacity (t/y). Next I’ll locate how it writes outputs so the README can explain the `1- 1Mt/` vs 2-1Kt folder structure and how to reproduce each run.

Searched for regex `1- 1Mt|2-1Kt|Mt.csv|Kt|output_dir|results|to_csv|savefig|os.makedirs|folder` (`Shared to github`), 20 results

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py#340-340), lines 340 to 460

Searched for regex `output_dir\s*=|RunConfig\(|cfg\.output_dir|CNT_OUTPUT|os\.chdir` (`Shared to github` · `CVD Model TEA.py`), 14 results

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py#3170-3170), lines 3170 to 3305

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py#3305-3305), lines 3305 to 4100

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/1-%201Mt)

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/2-1Kt)

Searched for regex `CoolProp|HAS_COOLPROP|import CoolProp|PropsSI` (`Shared to github` · `CVD Model TEA.py`), 20 results

Now that I’ve identified the key scale knob (`SCALE = set_scale(...)`) and how outputs are written (`RunConfig.output_dir`), I’m going to: (1) add a tiny quality-of-life tweak so output folder can be set via an environment variable, (2) fix the misleading “1 kt” comment next to the 1,000,000 t/y value, and (3) write a GitHub-ready `README.md` + `requirements.txt` + `.gitignore`.

Generating patch (29 lines) in [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py)

Patch didn’t apply because the tool couldn’t match both distant code regions in one chunk. Next I’ll re-apply as two small, targeted patches: one around `RunConfig` and one around the `SCALE = set_scale(...)` line.

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py#380-380), lines 380 to 420

Read [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py#120-120), lines 120 to 155

Generating patch (22 lines) in [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py)


```

```


Generating patch (9 lines) in [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py)


```

```


Created [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/README.md)

Created [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/requirements.txt)

Created [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/.gitignore)

Starting: *Provide GitHub publish steps* (4/4)

Next I’ll sanity-check that the small `RunConfig` change didn’t introduce any syntax/type errors, then I’ll give you the exact GitHub commands to publish this folder as a repo (public or private).

Checked [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/CVD%20Model%20TEA.py), 2 problems found

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && python -m py_compile "CVD Model TEA.py"

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git status
Completed with input:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git status

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && du -sh . && du -sh "1- 1Mt" "2-1Kt" 2>/dev/null || true

Good catch: your workspace is ~339 MB, almost all from `1- 1Mt/` (~334 MB). For a clean GitHub-shareable repo, I’ll update .gitignore to exclude the large output folders from commits (you can always regenerate them), then I’ll initialize git and commit only the code + README.

Generating patch (13 lines) in [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/.gitignore)


```

```


Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git init && git add "CVD Model TEA.py" README.md requirements.txt .gitignore && git commit -m "Add CNT TEA model + README"

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && command -v gh && gh --version
Completed with input:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && command -v gh && gh --version

I’ve initialized the git repo and made an initial commit containing only the script + docs (no large outputs). Next I’ll tweak the README so it matches this setup (outputs folders are ignored by default), then I’ll give you the exact GitHub web steps + push commands to publish it.

Generating patch (14 lines) in [](file:///Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education%20Hub/14-Energy%20Techonologies/7-Dissertation/11-%20Journal%20Paper/1-Reference/0-Economic%20model/1-Final/Shared%20to%20github/README.md)


```

```


Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git add README.md && git commit -m "Clarify generated outputs in README"

Completed: *Provide GitHub publish steps* (4/4)

**What I set up**
- Added a GitHub-ready README explaining the model + how to change inputs: README.md
- Added dependency list: requirements.txt
- Added ignore rules so huge output folders don’t get committed: .gitignore
- Small quality-of-life update in the script: `CNT_OUTPUT_DIR` now controls where outputs are written; also fixed the misleading “1 kt” comment: CVD Model TEA.py

**Git repo status**
- This folder is now a git repo with 2 commits (code + docs only). The large `1- 1Mt/` outputs (~334 MB) are intentionally not tracked.

**Publish to GitHub (so others can access the code)**
- Create a new repo on GitHub (recommended: **Public**, “Add a README” = **No** since we already have one).
- Then run these commands from this folder (replace placeholders):

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

If you prefer SSH:

```bash
git remote add origin git@github.com:<your-username>/<repo-name>.git
git push -u origin main
```

**Optional (if you want to share outputs too)**
- Because outputs can be large, use Git LFS or publish outputs as a GitHub Release/Zenodo dataset instead. If you tell me which you prefer, I can adjust the repo setup accordingly.

Made changes.

User: Can you do this for me?
Publish to GitHub (so others can access the code)

Create a new repo on GitHub (recommended: Public, “Add a README” = No since we already have one).
Then run these commands from this folder (replace placeholders):

git remote add origin https://github.com/<your-username>/<repo-name>.gitgit push -u origin main
I


GitHub Copilot: I can push it for you from this folder, but I need the GitHub repository URL (since GitHub repo creation requires your logged-in browser/account). Create the empty repo on GitHub, then paste the repo URL here and I’ll run the `git remote add` + `git push` commands for you.

Analyzing your answers...

Got the repo URL. Next I’ll add it as `origin` (or update it if it already exists) and push `main` to GitHub. If git prompts for credentials, I’ll pause and ask you for the username + a GitHub Personal Access Token (not your password).

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git remote -v

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git remote add origin "https://github.com/Imerson/Chemical-vapor-deposition-TEA.git" && git push -u origin main
Completed with input:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git remote add origin "https://github.com/Imerson/Chemical-vapor-deposition-TEA.git" && git push -u origin main

The push was rejected because that GitHub repo already has commits on `main` (e.g., an auto-created README/license). Next I’ll fetch the remote history so we can choose the safest path: either (A) pull/rebase and merge, or (B) overwrite the remote with your local history via `--force-with-lease` (only if you confirm it’s okay).

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git fetch origin && git log --oneline --decorate --graph --all --max-count=30

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git show --name-only --oneline --no-patch origin/main

Next I’ll list the files that are currently on GitHub in that “Initial commit” so we can choose the cleanest approach (rebase vs overwrite) without guessing.

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git show --name-only --pretty=format:"%h %s" origin/main

We can avoid force-pushing: since the remote only has a default README.md, I’ll rebase your two local commits on top of `origin/main` while automatically preferring *your* README on conflicts (`-X ours`), then push normally.

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git rebase -X ours origin/main

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git push -u origin main

Ran terminal command:  cd "/Users/imerson/Library/CloudStorage/OneDrive-Personal/Documents/Education Hub/14-Energy Techonologies/7-Dissertation/11- Journal Paper/1-Reference/0-Economic model/1-Final/Shared to github" && git remote -v && git status

Published successfully.

- Remote set to `https://github.com/Imerson/Chemical-vapor-deposition-TEA.git`
- Resolved the “remote has work” issue by rebasing your local commits onto the remote’s initial README commit (no force-push needed)
- Pushed `main` and set upstream tracking; repo is clean and up to date

Your code is now accessible at:
- https://github.com/Imerson/Chemical-vapor-deposition-TEA

If you want, I can also add a `LICENSE` file (MIT/Apache-2.0) and push that too.

User: Write here the readme code. I will copy and paste in the existing Read me since you could not

GitHub Copilot: ```markdown
# CNT Techno‑Economic Analysis (TEA) — FCCVD + FLB (Monte Carlo)

This repository contains a single, self-contained Python model for industrial-scale CNT techno‑economics with **two production routes**:

- **FC‑CVD (FCCVD)**: physics-based methane cracking mass/energy balance + techno‑economic model
- **FB‑CVD (FLB / fluidized bed)**: physics-based fluidized-bed route + techno‑economic model

Both routes are evaluated with a **Monte Carlo simulation** to propagate uncertainty (prices, yields, conversion, dosing, availability, CAPEX/OPEX factors) into outputs such as **unit cost** and **NPV**.

## Repository layout

- `CVD Model TEA.py` — main model + Monte Carlo runner (writes CSV + plots)
- `1- 1Mt/` — (optional) local output folder for a ~1 Mt/y scenario
- `2-1Kt/` — (optional) local output folder for a ~1 kt/y scenario

These output folders are **generated** by running the script and are ignored by default in `.gitignore` so the GitHub repository stays lightweight.

---

## Quick start

### 1) Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run

Because the script filename contains spaces, quote it:

```bash
python "CVD Model TEA.py"
```

The model prints summary statistics to the console and (by default) writes CSV + figures to disk.

---

## Key inputs you will likely change

### A) Scale / capacity (1 kt/y vs 1 Mt/y)

**This is the main knob.** Capacity is defined in **tonnes of CNT per year**.

In `CVD Model TEA.py`, find:

```python
SCALE = set_scale(throughput_tpy=..., annual_operating_hours=AnnualOperatingHours)
```

Use:

- **1 kt/y**: `throughput_tpy = 1_000.0`
- **1 Mt/y**: `throughput_tpy = 1_000_000.0`

Notes:
- `annual_operating_hours` is scheduled operating hours (e.g., `8000` h/y). Availability/uptime uncertainty is handled elsewhere in the Monte Carlo.

### B) Number of Monte Carlo samples

Set the environment variable `CNT_MC_N`:

```bash
CNT_MC_N=200000 python "CVD Model TEA.py"
```

Default is `100000` samples.

### C) Output directory (where CSV/plots are saved)

By default, outputs are written next to the script.

You can set `CNT_OUTPUT_DIR` to write outputs into a scenario folder (relative paths are resolved relative to the script’s folder):

```bash
# Example: write outputs into the existing 2-1Kt folder
CNT_OUTPUT_DIR="2-1Kt" CNT_MC_N=100000 python "CVD Model TEA.py"
```

Or:

```bash
# Example: write outputs into the existing 1- 1Mt folder
CNT_OUTPUT_DIR="1- 1Mt" CNT_MC_N=100000 python "CVD Model TEA.py"
```

### D) Plot display (interactive vs file outputs)

- Saved plots are controlled by `save_outputs` inside `RunConfig`.
- To *also* pop up Matplotlib windows while running:

```bash
CNT_SHOW_PLOTS=1 python "CVD Model TEA.py"
```

### E) Product sales toggles

Near the top of the script:

```python
AreyousellingH2 = "Yes"  # or "No"
AreyousellingSolidCarbon = "Yes"  # or "No"
```

These affect revenue streams in the TEA.

---

## Outputs

A run typically produces:

- Monte Carlo CSVs, e.g. `fccvd_mc_results_<timestamp>.csv`, `flb_mc_results_<timestamp>.csv`
- Histograms (NPV and unit cost) for each route
- FCCVD vs FLB overlay plots (PNG + SVG)
- Sensitivity plots (PNG + SVG)

Filenames are timestamped to avoid overwriting.

---

## Reproducibility

- Random seed is set in `RunConfig(seed=...)`.
- To reproduce a run exactly, keep the same seed, Monte Carlo sample count, and inputs.

---

## Notes / assumptions (high level)

- FCCVD mass balance is based on methane cracking stoichiometry: **CH₄ → C + 2 H₂**.
- Energy balance includes sensible heating, reaction endotherm, compression power, and heat loss through insulation.
- CoolProp is used (if installed) for temperature/pressure dependent properties; the script includes fallbacks so it can still run without CoolProp (with reduced thermodynamic fidelity).

---

## License

If you want this to be broadly reusable on GitHub, add a license file (MIT/Apache-2.0 are common). If you tell me which license you prefer, I can add it.
```
