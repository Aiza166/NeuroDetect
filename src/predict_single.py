"""Single-patient inference CLI.

Defaults to the HONEST model (UPDRS / MoCA / FunctionalAssessment excluded).
The leaky all-features baseline is available behind an explicit flag and warns
loudly about what it is.

    python src/predict_single.py                      # honest model, medians
    python src/predict_single.py --interactive        # prompt for each feature
    python src/predict_single.py --model leaky        # documented baseline
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

from src.data_pipeline.preprocess_data import LEAKY_COLS, load_raw  # noqa: E402

ARM = {"honest": "HONEST", "leaky": "ALL_FEATURES_LEAKY"}

DISCLAIMER = (
    "NOTE: demonstration of an ML pipeline on SYNTHETIC data. "
    "This is not a screening or diagnostic tool. Do not use it for any "
    "clinical purpose."
)

LEAKY_WARNING = """
!! You selected the ALL_FEATURES_LEAKY baseline.
!!
!! Its higher apparent accuracy (0.8504 vs 0.6912) is NOT better prediction.
!! It comes from UPDRS, MoCA and FunctionalAssessment -- clinical assessment
!! scores that already encode the diagnosis. A clinician records them because
!! they have already assessed the patient, so the model is restating a
!! diagnosis rather than predicting one.
!!
!! This model exists as a documented baseline. See ANALYSIS.md.
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=ARM, default="honest",
                   help="which arm to load (default: honest)")
    p.add_argument("--interactive", action="store_true",
                   help="prompt for each feature instead of using medians")
    return p.parse_args()


def load_arm(arm):
    model_path = f"models/parkinsons_model_{arm}.keras"
    scaler_path = f"models/scaler_{arm}.pkl"
    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run: python src/models/train_model.py")
    bundle = joblib.load(scaler_path)
    return load_model(model_path), bundle["scaler"], bundle["features"]


def collect(features, defaults, interactive):
    """Feature values, in the exact order the scaler was fitted on."""
    if not interactive:
        print("Using dataset medians for every feature "
              "(pass --interactive to enter values).")
        return {f: defaults[f] for f in features}

    print("Press Enter to accept the median shown in brackets.\n")
    values = {}
    for f in features:
        d = defaults[f]
        while True:
            raw = input(f"  {f} [{d:g}]: ").strip()
            if not raw:
                values[f] = d
                break
            try:
                values[f] = float(raw)
                break
            except ValueError:
                print("    not a number, try again")
    return values


def main():
    args = parse_args()
    arm = ARM[args.model]

    print(DISCLAIMER)
    if arm == "ALL_FEATURES_LEAKY":
        print(LEAKY_WARNING)

    model, scaler, features = load_arm(arm)
    print(f"Model: {arm} ({len(features)} features)")
    if arm == "HONEST":
        print(f"Excluded as target leakage: {', '.join(LEAKY_COLS)}")
    print()

    X_raw, _ = load_raw()
    defaults = X_raw.median().to_dict()

    values = collect(features, defaults, args.interactive)
    # Column order must match the scaler's fit order exactly.
    input_df = pd.DataFrame([[values[f] for f in features]], columns=features)

    prob = float(model.predict(scaler.transform(input_df), verbose=0)[0][0])
    label = "Parkinson's indicated" if prob >= 0.50 else "Parkinson's not indicated"

    print(f"\nProbability: {prob:.4f}  (threshold 0.50)")
    print(f"Result:      {label}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
