"""Supplementary CatBoost comparison for manuscript applsci-4663183.

This is a frozen, exploratory extension of the original nested evaluation.
It never modifies original model outputs and never publishes individual records.
Run ``python catboost_extension.py check`` before ``... run``.  Defaults assume
this script is placed beside analyze.py in the public reproducibility package.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CLASSES = np.arange(10)
MODEL_SEED = SPLIT_SEED = 20261014
FEATURES = ["age_years", "sex", "education"] + [f"emp{k}" for k in range(1, 8)] + ["p15", "p18", "p19", "p20", "p21"]
CATEGORICAL = ["sex", "education"]
CLASS_LABELS = ["trust", "knowledge_skills", "privacy_data", "jobs_unfair_decisions", "errors_quality", "rules_accountability", "cost", "device_internet_access", "no_need", "hard_to_say"]
GRID = [dict(iterations=i, depth=d, l2_leaf_reg=l) for i, d, l in itertools.product((100, 500), (4, 6), (3, 10))]
FIXED_PARAMS = dict(
    loss_function="MultiClass", eval_metric="MultiClass", learning_rate=0.05,
    boosting_type="Ordered", task_type="CPU", random_seed=MODEL_SEED,
    thread_count=1, use_best_model=False, allow_writing_files=False,
    verbose=False, classes_count=10, grow_policy="SymmetricTree",
    bootstrap_type="Bayesian", bagging_temperature=1.0, random_strength=1.0,
    one_hot_max_size=2,
)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path, frame, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_data(path):
    data = pd.read_csv(path)
    required = FEATURES + ["site", "target"]
    if len(data) != 3528 or not set(required).issubset(data):
        raise ValueError("Expected the original 3528-row harmonized data and fixed predictor columns.")
    if data[required].isna().any().any():
        raise ValueError("The fixed analytic variables must be complete.")
    if not np.array_equal(np.sort(data.target.unique()), CLASSES):
        raise ValueError("Expected target codes 0 through 9.")
    data = data[required].copy().reset_index(drop=True)
    data["site"] = data.site.astype(str)
    data["target"] = data.target.astype(int)
    if data.site.nunique() != 6:
        raise ValueError("Expected the same six supplied sample labels.")
    for column in CATEGORICAL:
        values = data[column].to_numpy()
        if not np.equal(values, values.astype(int)).all():
            raise ValueError(f"Expected integer category codes in {column}.")
        data[column] = data[column].astype(int).astype(str)
    return data


def split_hash(train, test):
    return digest_bytes(np.asarray(train, dtype="<i8").tobytes() + b"|" + np.asarray(test, dtype="<i8").tobytes())


def make_splits(data):
    tasks = []
    for j, site in enumerate(sorted(data.site.unique()), 1):
        train = np.flatnonzero(data.site.to_numpy() != site)
        test = np.flatnonzero(data.site.to_numpy() == site)
        tasks.append(dict(regime="loso", fold=j, sample=site, train=train, test=test, split_sha256=split_hash(train, test)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    strata = data.site.astype(str) + "_" + data.target.astype(str)
    for j, (train, test) in enumerate(cv.split(data, strata), 1):
        tasks.append(dict(regime="random", fold=j, sample=None, train=train, test=test, split_sha256=split_hash(train, test)))
    for regime in ("loso", "random"):
        rows = [t["test"] for t in tasks if t["regime"] == regime]
        if not np.array_equal(np.sort(np.concatenate(rows)), np.arange(len(data))):
            raise AssertionError("Outer tests do not partition the original records exactly once.")
    return tasks


def priors_for_training(train):
    pooled = (np.bincount(train.target, minlength=10) + 1) / (len(train) + 10)
    individual = []
    for site in sorted(train.site.unique()):
        y = train.loc[train.site.eq(site), "target"].to_numpy()
        individual.append((np.bincount(y, minlength=10) + 1) / (len(y) + 10))
    return pooled, np.mean(individual, axis=0)


def validate_probabilities(p, n):
    if p.shape != (n, 10) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise AssertionError("Invalid ten-category probability matrix.")


def make_model(candidate):
    return CatBoostClassifier(**FIXED_PARAMS, **candidate, cat_features=CATEGORICAL)


def ece(y, p):
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bins = np.minimum((confidence * 10).astype(int), 9)
    value = 0.0
    for k in range(10):
        selected = bins == k
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def metrics(y, p):
    pred = p.argmax(axis=1)
    return dict(n=len(y), log_loss=float(log_loss(y, p, labels=CLASSES)),
                brier=float(np.mean(np.sum((p - np.eye(10)[y]) ** 2, axis=1))),
                accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
                macro_recall=float(recall_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
                ece=ece(y, p))


def protocol_definition(data_path, tasks):
    return dict(
        schema_version=1,
        study="applsci-4663183",
        analysis_status="Exploratory analysis added during revision after original-model results were known; not preregistered.",
        data_sha256=digest_file(data_path), n=3528,
        implementation_sha256=digest_file(__file__),
        features=FEATURES, categorical_features=CATEGORICAL, target_classes=CLASSES.tolist(),
        fixed_parameters=FIXED_PARAMS, candidate_grid=GRID,
        outer_validation="Six complete-site holdouts and original five jointly site-by-target stratified random folds.",
        inner_selection="Leave one training site out; minimize unweighted mean of site log losses; ties use the first grid entry.",
        training_weights="No class balancing or explicit sample weights; equal-site weighting is applied in selection and evaluation.",
        preprocessing="Only fixed features; sex and education integer codes converted to strings. All CatBoost quantization and categorical transformations fit within each training subset. No shared quantized Pool or eval_set.",
        calibration=dict(method="temperature scaling", bounds=[0.25, 4.0], optimizer="bounded minimize_scalar over log(T)", clip=1e-15, criterion="equal-site-weighted selected inner out-of-fold log loss"),
        model_seed=MODEL_SEED, split_seed=SPLIT_SEED,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__, catboost=catboost.__version__),
        outer_splits=[dict(regime=t["regime"], fold=t["fold"], sample=t["sample"], n_train=len(t["train"]), n_test=len(t["test"]), split_sha256=t["split_sha256"]) for t in tasks],
        reporting="All raw and calibrated results reported regardless of comparative performance. Original model outputs remain unchanged.",
    )


def freeze_protocol(output, definition):
    path = Path(output) / "protocol.json"
    fingerprint = digest_bytes(canonical_json(definition).encode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("protocol_sha256") != fingerprint or existing.get("definition") != definition:
            raise RuntimeError("Existing protocol differs in code, data, software, or settings. Use a separate output folder; do not silently resume.")
    else:
        atomic_json(path, dict(recorded_before_first_fit_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=fingerprint, definition=definition))
    return fingerprint


def checkpoint_paths(task, output, private_output):
    stem = f"{task['regime']}_fold_{task['fold']}"
    return Path(output) / "checkpoints" / (stem + ".json"), Path(private_output) / "checkpoints" / (stem + ".npz")


def load_checkpoint(task, output, private_output, fingerprint):
    meta_path, arrays_path = checkpoint_paths(task, output, private_output)
    if not meta_path.exists() and not arrays_path.exists():
        return None
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as z:
            if str(z["protocol_sha256"].item()) != fingerprint:
                raise RuntimeError(f"Incompatible private checkpoint: {arrays_path.name}")
    if not meta_path.exists() or not arrays_path.exists():
        return None  # Interrupted atomic pair; recompute this fold under the same protocol.
    meta = json.loads(meta_path.read_text())
    if meta.get("protocol_sha256") != fingerprint or meta.get("split_sha256") != task["split_sha256"] or meta.get("private_checkpoint_sha256") != digest_file(arrays_path):
        raise RuntimeError(f"Invalid checkpoint identity or digest: {meta_path.name}")
    with np.load(arrays_path, allow_pickle=False) as z:
        if not np.array_equal(z["test_indices"], task["test"]):
            raise RuntimeError("Checkpoint test row order differs from the protected split.")
        for name in ("catboost", "catboost_temp", "prior", "prior_equal_site"):
            validate_probabilities(z[name], len(task["test"]))
    return meta


def fit_outer(task, data, output, private_output, fingerprint):
    existing = load_checkpoint(task, output, private_output, fingerprint)
    if existing is not None:
        return dict(regime=task["regime"], fold=task["fold"], resumed=True)
    started = time.perf_counter()
    train = data.iloc[task["train"]].reset_index(drop=True)
    test = data.iloc[task["test"]].reset_index(drop=True)
    groups = train.site.to_numpy()
    y = train.target.to_numpy()
    candidates = []
    best = None
    with threadpool_limits(limits=1):
        for candidate_index, candidate in enumerate(GRID):
            candidate_started = time.perf_counter()
            oof = np.full((len(train), 10), np.nan)
            site_losses = []
            for site in sorted(set(groups)):
                va = groups == site
                tr = ~va
                model = make_model(candidate)
                # NO outer test or inner validation is passed as eval_set.
                model.fit(train.loc[tr, FEATURES], y[tr])
                if not np.array_equal(model.classes_, CLASSES) or model.tree_count_ != candidate["iterations"]:
                    raise AssertionError("Unexpected classes or early stopping.")
                p = model.predict_proba(train.loc[va, FEATURES])
                validate_probabilities(p, int(va.sum()))
                oof[va] = p
                site_losses.append(dict(site=str(site), log_loss=float(log_loss(y[va], p, labels=CLASSES))))
            score = float(np.mean([x["log_loss"] for x in site_losses]))
            candidates.append(dict(candidate_index=candidate_index, parameters=candidate, inner_log_loss=score, inner_site_losses=site_losses))
            if best is None or score < best[0]:
                best = (score, candidate_index, candidate.copy(), oof.copy())
            print(json.dumps(dict(event="candidate_complete", regime=task["regime"], fold=task["fold"], candidate_index=candidate_index, candidate_count=len(GRID), elapsed_seconds=time.perf_counter()-candidate_started)), flush=True)
        score, candidate_index, candidate, oof = best
        validate_probabilities(oof, len(train))
        weights = np.array([1.0 / np.sum(groups == s) for s in groups])
        logp = np.log(np.clip(oof, 1e-15, 1))

        def objective(log_temperature):
            p = softmax(logp / np.exp(log_temperature), axis=1)
            return float(np.average(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)), weights=weights))

        opt = minimize_scalar(objective, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
        if not opt.success or not np.isfinite(opt.fun):
            raise RuntimeError("Temperature estimation did not converge.")
        temperature = float(np.exp(opt.x))
        model = make_model(candidate)
        model.fit(train[FEATURES], y)
        if not np.array_equal(model.classes_, CLASSES) or model.tree_count_ != candidate["iterations"]:
            raise AssertionError("Unexpected outer-refit classes or tree count.")
        raw = model.predict_proba(test[FEATURES])
        calibrated = softmax(np.log(np.clip(raw, 1e-15, 1)) / temperature, axis=1)
        validate_probabilities(raw, len(test))
        validate_probabilities(calibrated, len(test))
        if not np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1)):
            raise AssertionError("Positive scalar temperature unexpectedly changed top labels.")
        pooled, equal_site = priors_for_training(train)
        arrays = dict(catboost=raw, catboost_temp=calibrated,
                      prior=np.broadcast_to(pooled, raw.shape).copy(),
                      prior_equal_site=np.broadcast_to(equal_site, raw.shape).copy(),
                      test_indices=task["test"], protocol_sha256=np.array(fingerprint))
        meta_path, arrays_path = checkpoint_paths(task, output, private_output)
        atomic_npz(arrays_path, **arrays)
        meta = dict(protocol_sha256=fingerprint, regime=task["regime"], fold=task["fold"], sample=task["sample"],
                    split_sha256=task["split_sha256"], n_train=len(train), n_test=len(test),
                    selected_candidate_index=candidate_index, selected_parameters=candidate,
                    inner_log_loss=score, temperature=temperature, temperature_inner_loss=float(opt.fun),
                    candidates=candidates, fitted_parameters=model.get_all_params(),
                    elapsed_seconds=time.perf_counter() - started,
                    private_checkpoint_sha256=digest_file(arrays_path))
        atomic_json(meta_path, meta)
    return dict(regime=task["regime"], fold=task["fold"], resumed=False, elapsed_seconds=meta["elapsed_seconds"])


def summarize_predictions(data, predictions, regime):
    metric_rows, class_rows, confusions = [], [], {}
    y = data.target.to_numpy()
    for model, p in predictions.items():
        validate_probabilities(p, len(data))
        per_site = []
        for site in ["Pooled"] + sorted(data.site.unique()):
            ix = np.ones(len(data), dtype=bool) if site == "Pooled" else data.site.eq(site).to_numpy()
            row = dict(regime=regime, model=model, site=site, **metrics(y[ix], p[ix]))
            metric_rows.append(row)
            if site != "Pooled":
                per_site.append(row)
            cm = confusion_matrix(y[ix], p[ix].argmax(axis=1), labels=CLASSES)
            if site == "Pooled":
                confusions[model] = cm
            support = cm.sum(axis=1)
            selected = cm.sum(axis=0)
            for k, label in enumerate(CLASS_LABELS):
                class_rows.append(dict(regime=regime, model=model, site=site, class_index=k,
                                       category=label, support=int(support[k]), predicted_count=int(selected[k]),
                                       true_positive=int(cm[k, k]), recall=float(cm[k, k] / support[k]) if support[k] else 0.0,
                                       precision=float(cm[k, k] / selected[k]) if selected[k] else 0.0,
                                       observed_proportion=float(support[k] / ix.sum()),
                                       mean_predicted_probability=float(p[ix, k].mean())))
        metric_rows.append(dict(regime=regime, model=model, site="Equal-site mean", n=len(data),
                                **{key:float(np.mean([row[key] for row in per_site])) for key in metrics(y, p) if key != "n"}))
    return metric_rows, class_rows, confusions


def baseline_check(data, tasks, baseline_results, baseline_private):
    status = []
    for regime in ("loso", "random"):
        prior = np.full((len(data), 10), np.nan)
        for task in tasks:
            if task["regime"] == regime:
                prior[task["test"]] = priors_for_training(data.iloc[task["train"]])[0]
        rows, _, _ = summarize_predictions(data, {"prior":prior}, regime)
        actual = pd.DataFrame(rows).set_index("site")
        path = Path(baseline_results) / f"metrics_{regime}_full.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Original aggregate metrics required: {path}")
        expected = pd.read_csv(path).query("model == 'prior'").set_index("site")
        for site, row in actual.iterrows():
            for metric in ("n", "log_loss", "brier", "accuracy", "macro_f1", "macro_recall", "ece"):
                if not np.isclose(row[metric], expected.loc[site, metric], rtol=0, atol=1e-12):
                    raise AssertionError(f"Original prior mismatch: {regime}/{site}/{metric}")
        prediction_file = Path(baseline_private) / f"predictions_{regime}_full.npz"
        private_verified = False
        if prediction_file.is_file():
            with np.load(prediction_file, allow_pickle=False) as z:
                if z["prior"].shape != prior.shape or not np.allclose(z["prior"], prior, rtol=0, atol=1e-15):
                    raise AssertionError("Original private prior predictions differ; check original row order.")
                private_verified = True
        status.append(dict(regime=regime, original_prior_aggregate_match=True, original_private_prior_available_and_matching=private_verified))
    return status


def assemble(data, tasks, output, private_output, fingerprint, baseline_results, baseline_private):
    output, private_output = Path(output), Path(private_output)
    all_metrics, all_classes, all_configs, comparisons, gains = [], [], [], [], []
    for regime in ("loso", "random"):
        predictions = {name:np.full((len(data), 10), np.nan) for name in ("prior", "prior_equal_site", "catboost", "catboost_temp")}
        for task in tasks:
            if task["regime"] != regime:
                continue
            meta = load_checkpoint(task, output, private_output, fingerprint)
            if meta is None:
                raise RuntimeError("Cannot summarize an incomplete outer evaluation.")
            all_configs.append(meta)
            _, path = checkpoint_paths(task, output, private_output)
            with np.load(path, allow_pickle=False) as z:
                for name in predictions:
                    predictions[name][task["test"]] = z[name]
        atomic_npz(private_output / f"predictions_{regime}_catboost.npz", protocol_sha256=np.array(fingerprint), **predictions)
        metric_rows, class_rows, confusions = summarize_predictions(data, predictions, regime)
        all_metrics.extend(metric_rows)
        all_classes.extend(class_rows)
        metric_frame = pd.DataFrame(metric_rows)
        atomic_csv(output / f"metrics_{regime}_catboost.csv", metric_frame, index=False)
        for name, matrix in confusions.items():
            atomic_csv(output / f"confusion_{regime}_{name}.csv", pd.DataFrame(matrix), index_label="actual_class")
        old = pd.read_csv(Path(baseline_results) / f"metrics_{regime}_full.csv")
        old.insert(0, "regime", regime)
        comparisons.append(pd.concat([old, metric_frame.loc[metric_frame.model.ne("prior")]], ignore_index=True))
        indexed = metric_frame.set_index(["model", "site"])
        for prior_name in ("prior", "prior_equal_site"):
            for model in ("catboost", "catboost_temp"):
                for site in ["Pooled", "Equal-site mean"] + sorted(data.site.unique()):
                    b = float(indexed.loc[(prior_name, site), "log_loss"])
                    loss = float(indexed.loc[(model, site), "log_loss"])
                    gains.append(dict(regime=regime, baseline=prior_name, model=model, site=site, prior_log_loss=b, model_log_loss=loss, log_loss_gain=b-loss))
    gains_frame = pd.DataFrame(gains)
    decompositions = []
    for prior_name in ("prior", "prior_equal_site"):
        for model in ("catboost", "catboost_temp"):
            for site in ["Pooled", "Equal-site mean"] + sorted(data.site.unique()):
                selected = gains_frame.loc[(gains_frame.baseline == prior_name) & (gains_frame.model == model) & (gains_frame.site == site)].set_index("regime")
                a, b = selected.loc["loso"], selected.loc["random"]
                raw_gap = a.model_log_loss-b.model_log_loss
                prior_gap = a.prior_log_loss-b.prior_log_loss
                gain_gap = b.log_loss_gain-a.log_loss_gain
                if not np.isclose(raw_gap, prior_gap+gain_gap, rtol=0, atol=1e-14):
                    raise AssertionError("Arithmetic gap decomposition failed.")
                decompositions.append(dict(baseline=prior_name, model=model, site=site, loso_model_loss=a.model_log_loss, random_model_loss=b.model_log_loss, raw_loss_gap=raw_gap, prior_loss_gap=prior_gap, random_minus_loso_gain=gain_gap))
    atomic_csv(output / "metrics_all.csv", pd.DataFrame(all_metrics), index=False)
    atomic_csv(output / "model_comparison.csv", pd.concat(comparisons, ignore_index=True), index=False)
    atomic_csv(output / "classwise_performance.csv", pd.DataFrame(all_classes), index=False)
    atomic_csv(output / "model_gains_by_design.csv", gains_frame, index=False)
    atomic_csv(output / "loss_gap_decomposition.csv", pd.DataFrame(decompositions), index=False)
    atomic_json(output / "selected_configurations.json", all_configs)
    checks = baseline_check(data, tasks, baseline_results, baseline_private)
    atomic_json(output / "validation.json", dict(protocol_sha256=fingerprint, n=3528, outer_folds=len(tasks), all_probability_matrices_complete=True, original_baseline_checks=checks,
                                               metric_aggregation="Pooled and equal-site summaries are distinct; all sites have equal weight only in the latter.",
                                               inference_limit="These outputs are descriptive. No claim of independent confirmation, population superiority, or comprehensive tuning is made."))
    return pd.concat(comparisons, ignore_index=True).query("site == 'Equal-site mean'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "run", "assemble"])
    parser.add_argument("--data", type=Path, default=ROOT / "private/analysis_data.csv")
    parser.add_argument("--baseline-results", type=Path, default=ROOT / "results")
    parser.add_argument("--baseline-private", type=Path, default=ROOT / "private")
    parser.add_argument("--output", type=Path, default=ROOT / "results/revision/catboost")
    parser.add_argument("--private-output", type=Path, default=ROOT / "private/catboost")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.output.resolve() == args.baseline_results.resolve() or args.private_output.resolve() == args.baseline_private.resolve():
        parser.error("Extension outputs must be separate from original outputs")
    data = load_data(args.data)
    tasks = make_splits(data)
    checks = baseline_check(data, tasks, args.baseline_results, args.baseline_private)
    definition = protocol_definition(args.data, tasks)
    if args.command == "check":
        print(json.dumps(dict(status="passed_without_fitting", n=len(data), predictors=len(FEATURES), candidates=len(GRID), outer_folds=len(tasks), expected_inner_fits=480, expected_outer_refits=11, baseline_checks=checks), indent=2))
        return
    fingerprint = freeze_protocol(args.output, definition)
    if args.command == "run":
        pending = [task for task in tasks if load_checkpoint(task, args.output, args.private_output, fingerprint) is None]
        print(f"Frozen protocol {fingerprint}; {len(pending)} outer folds pending.", flush=True)
        if args.workers == 1:
            for task in pending:
                print(json.dumps(fit_outer(task, data, args.output, args.private_output, fingerprint)), flush=True)
        elif pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = [executor.submit(fit_outer, task, data, args.output, args.private_output, fingerprint) for task in pending]
                for future in as_completed(futures):
                    print(json.dumps(future.result()), flush=True)
    summary = assemble(data, tasks, args.output, args.private_output, fingerprint, args.baseline_results, args.baseline_private)
    print(summary[["regime", "model", "log_loss", "accuracy", "macro_f1", "macro_recall"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
