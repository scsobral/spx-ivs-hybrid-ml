#!/usr/bin/env python3
"""
Test Pipeline Reproducibility & Contract Smoke Test
===================================================
Master's Thesis: Study of Implied Volatility Surface with Machine Learning Techniques
Author: Sérgio da Cunha Sobral | Supervisor: Profª. Drª. Diana Mendes (ISCTE-IUL)

This test executes an end-to-end dry-run validation of the entire quantitative
modeling architecture on real CBOE SPX option slices. It tests:
  1. Auxiliary functions and evaluation engine (RMSE, MAE, R2, MedAE, Bias, 5x5 matrices)
  2. Parquet data ingestion across all 4 splits
  3. All 7 model pipelines:
     - OLS
     - 5th-Degree Polynomial
     - Cubic B-Splines (TensorBSplineFeatures)
     - Random Forest Regressor
     - XGBoost Regressor
     - Hybrid Splines + Random Forest
     - Hybrid Splines + XGBoost
  4. 3D surface meshgrid generation & heatmap visualization
  5. Integrity audit of existing metadata JSONs (22 files) and Excel tables (12 files)
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

import pandas as pd
import openpyxl

# Force non-interactive backend for headless execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Scikit-learn & ML imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Add utils to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "03_notebooks", "00_utils"))
import auxiliary_functions as af


class TensorBSplineFeatures(BaseEstimator, TransformerMixin):
    """
    Tensor-Product B-Spline Transformer for (log_moneyness, T):
      - Standardizes each dimension (z-score)
      - Computes cubic 1D B-spline bases
      - Returns the tensor product B_m ⊗ B_T
    """
    def __init__(
        self,
        n_knots_m: int = 5,
        n_knots_t: int = 5,
        degree: int = 3,
        include_bias: bool = False,
        knots_mode: str = "quantile",
        extrapolation: str = "linear",
        with_scaler: bool = True,
    ):
        self.n_knots_m = n_knots_m
        self.n_knots_t = n_knots_t
        self.degree = degree
        self.include_bias = include_bias
        self.knots_mode = knots_mode
        self.extrapolation = extrapolation
        self.with_scaler = with_scaler

    def _fit_single(self, x: np.ndarray, n_knots: int):
        scaler = None
        x_fit = x
        if self.with_scaler:
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x)
        try:
            spline = SplineTransformer(
                n_knots=n_knots,
                degree=self.degree,
                include_bias=self.include_bias,
                extrapolation=self.extrapolation,
                knots=self.knots_mode,
            )
            B = spline.fit_transform(x_fit)
        except TypeError:
            spline = SplineTransformer(
                n_knots=n_knots,
                degree=self.degree,
                include_bias=self.include_bias,
                extrapolation=self.extrapolation,
            )
            B = spline.fit_transform(x_fit)
        return scaler, spline, B.shape[1]

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=float)
        m = X[:, [0]]
        t = X[:, [1]]
        self.scaler_m_, self.spline_m_, self.n_out_m_ = self._fit_single(m, self.n_knots_m)
        self.scaler_t_, self.spline_t_, self.n_out_t_ = self._fit_single(t, self.n_knots_t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        m = X[:, [0]]
        t = X[:, [1]]
        m_in = self.scaler_m_.transform(m) if self.scaler_m_ is not None else m
        t_in = self.scaler_t_.transform(t) if self.scaler_t_ is not None else t
        Bm = self.spline_m_.transform(m_in)
        Bt = self.spline_t_.transform(t_in)
        # Tensor product (row-wise Kronecker)
        N, p = Bm.shape
        _, q = Bt.shape
        return (Bm[:, :, None] * Bt[:, None, :]).reshape(N, p * q)


class TestIVSPipelineReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_ivs_pipeline_")
        cls.temp_graficos = os.path.join(cls.temp_dir, "figures")
        cls.temp_tabelas = os.path.join(cls.temp_dir, "tables")
        os.makedirs(cls.temp_graficos, exist_ok=True)
        os.makedirs(cls.temp_tabelas, exist_ok=True)

        # Locate processed data
        cls.data_dir = os.path.join(BASE_DIR, "02_data", "processed")
        if not os.path.exists(cls.data_dir):
            cls.data_dir = os.path.join(BASE_DIR, "_legacy_pt", "02. Dados", "02. Dados Tratados")

        print("\n" + "=" * 70)
        print("  STARTING QUANTITATIVE PIPELINE REPRODUCIBILITY DRY-RUN")
        print("=" * 70)
        print(f"Base Directory: {BASE_DIR}")
        print(f"Data Source:    {cls.data_dir}")
        print(f"Temp Output:    {cls.temp_dir}\n")

        # Load real data samples
        cls.df_train_full = pd.read_parquet(os.path.join(cls.data_dir, "df_train_full.parquet")).head(1000).copy()
        cls.df_test_puro = pd.read_parquet(os.path.join(cls.data_dir, "df_test_puro.parquet")).head(500).copy()
        cls.df_inner_train = pd.read_parquet(os.path.join(cls.data_dir, "df_inner_train.parquet")).head(500).copy()
        cls.df_inner_tunning = pd.read_parquet(os.path.join(cls.data_dir, "df_inner_tunning.parquet")).head(500).copy()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        print("\n" + "=" * 70)
        print("  CLEANED UP TEMPORARY TESTING DIRECTORY")
        print("=" * 70)

    def test_01_data_splits_schema_and_types(self):
        """Verify that Parquet splits contain all required columns and valid numeric ranges."""
        print("Test 01: Verifying data splits schema and numeric properties...")
        req_cols = ["log_moneyness", "T", "option_is_call", "iv_teorica", "quote_date"]
        for name, df in [
            ("train_full", self.df_train_full),
            ("test_puro", self.df_test_puro),
            ("inner_train", self.df_inner_train),
            ("inner_tunning", self.df_inner_tunning),
        ]:
            for col in req_cols:
                self.assertIn(col, df.columns, f"Missing column {col} in {name}")
            self.assertFalse(df["iv_teorica"].isna().any(), f"NaNs found in iv_teorica in {name}")
            self.assertTrue((df["iv_teorica"] > 0).all(), f"Non-positive IV in {name}")
            self.assertTrue((df["T"] > 0).all(), f"Non-positive maturity T in {name}")
        print("  [OK] All 4 Parquet splits conform to expected schema and ranges.")

    def test_02_auxiliary_functions_and_styles(self):
        """Verify color swatches, line styles, and evaluation function exports."""
        print("Test 02: Checking auxiliary functions module integrity...")
        self.assertTrue(hasattr(af, "run_evaluations"))
        self.assertTrue(hasattr(af, "run_avaliacoes"))
        self.assertTrue(hasattr(af, "plot_ivs_and_variance_surfaces"))
        self.assertTrue(hasattr(af, "COLOR_PALETTE"))
        self.assertTrue(hasattr(af, "LINE_STYLES"))
        self.assertGreater(len(af.COLOR_PALETTE), 0)
        self.assertGreater(len(af.LINE_STYLES), 0)
        print("  [OK] Auxiliary functions and styling palettes intact.")

    def test_03_model_ols(self):
        """Fit OLS model on train_full slice and evaluate on test_puro."""
        print("Test 03: Testing Model 1 (OLS Linear Regression)...")
        features = ["log_moneyness", "T"]
        X_tr = self.df_train_full[features].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_te = self.df_test_puro[features].to_numpy(dtype=float)

        model = LinearRegression().fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        self.assertEqual(len(y_pred), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_pred).all())

        df_test = self.df_test_puro.copy()
        df_test["ivs_modelo"] = y_pred
        df_train = self.df_train_full.copy()
        df_train["ivs_modelo"] = model.predict(X_tr)

        # Run evaluation pipeline
        res = af.run_evaluations(
            df_inner_train=df_train.head(500),
            df_inner_tunning=df_train.head(500),
            df_train_full=df_train,
            df_test_puro=df_test,
            model="OLS_Test",
            DIR_TABELAS=self.temp_tabelas,
            DIR_GRAFICOS=self.temp_graficos,
            display_in_notebook=False,
        )
        self.assertTrue(os.path.exists(res["excel_path"]))
        self.assertGreater(res["globais_rows"], 0)
        print("  [OK] OLS pipeline, metrics calculation, and Excel export verified.")

    def test_04_model_polynomial(self):
        """Fit 5th-Degree Polynomial Regression pipeline."""
        print("Test 04: Testing Model 2 (5th-Degree Polynomial Pipeline)...")
        features = ["log_moneyness", "T"]
        X_tr = self.df_train_full[features].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_te = self.df_test_puro[features].to_numpy(dtype=float)

        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        self.assertEqual(len(y_pred), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_pred).all())
        print("  [OK] 5th-Degree Polynomial regression pipeline verified.")

    def test_05_model_splines(self):
        """Fit Tensor-Product Cubic B-Splines pipeline."""
        print("Test 05: Testing Model 3 (Tensor-Product Cubic B-Splines)...")
        features = ["log_moneyness", "T"]
        X_tr = self.df_train_full[features].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_te = self.df_test_puro[features].to_numpy(dtype=float)

        pipe = Pipeline([
            ("tensor_spline", TensorBSplineFeatures(n_knots_m=5, n_knots_t=5, degree=3)),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 7))),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        self.assertEqual(len(y_pred), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_pred).all())
        print("  [OK] Tensor-Product Cubic B-Splines pipeline verified.")

    def test_06_model_random_forest(self):
        """Fit mini Random Forest regressor with option_is_call feature."""
        print("Test 06: Testing Model 4 (Random Forest Regressor)...")
        features = ["log_moneyness", "T", "option_is_call"]
        X_tr = self.df_train_full[features].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_te = self.df_test_puro[features].to_numpy(dtype=float)

        rf = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=42)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)

        self.assertEqual(len(y_pred), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_pred).all())
        print("  [OK] Random Forest regressor pipeline verified.")

    def test_07_model_xgboost(self):
        """Fit mini XGBoost regressor."""
        print("Test 07: Testing Model 5 (XGBoost Regressor)...")
        features = ["log_moneyness", "T", "option_is_call"]
        X_tr = self.df_train_full[features].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_te = self.df_test_puro[features].to_numpy(dtype=float)

        model = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        self.assertEqual(len(y_pred), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_pred).all())
        print("  [OK] XGBoost regressor pipeline verified.")

    def test_08_model_hybrid_splines_rf(self):
        """Fit Hybrid Model 1: Splines baseline + Random Forest on residuals."""
        print("Test 08: Testing Model 6 (Hybrid: B-Splines + Random Forest on Residuals)...")
        features_spline = ["log_moneyness", "T"]
        features_tree = ["log_moneyness", "T", "option_is_call"]

        X_tr_spline = self.df_train_full[features_spline].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_tr_tree = self.df_train_full[features_tree].to_numpy(dtype=float)

        X_te_spline = self.df_test_puro[features_spline].to_numpy(dtype=float)
        X_te_tree = self.df_test_puro[features_tree].to_numpy(dtype=float)

        # 1. Spline baseline
        spline_pipe = Pipeline([
            ("tensor_spline", TensorBSplineFeatures(n_knots_m=5, n_knots_t=5, degree=3)),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 7))),
        ])
        spline_pipe.fit(X_tr_spline, y_tr)
        y_hat_spline_tr = spline_pipe.predict(X_tr_spline)
        y_hat_spline_te = spline_pipe.predict(X_te_spline)

        # 2. Residuals modeling
        residuals_tr = y_tr - y_hat_spline_tr
        rf_resid = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=42)
        rf_resid.fit(X_tr_tree, residuals_tr)
        e_hat_te = rf_resid.predict(X_te_tree)

        # 3. Hybrid recombination
        y_hat_hybrid = y_hat_spline_te + e_hat_te

        self.assertEqual(len(y_hat_hybrid), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_hat_hybrid).all())
        print("  [OK] Hybrid Splines + Random Forest pipeline verified.")

    def test_09_model_hybrid_splines_xgboost(self):
        """Fit Hybrid Model 2: Splines baseline + XGBoost on residuals."""
        print("Test 09: Testing Model 7 (Hybrid: B-Splines + XGBoost on Residuals)...")
        features_spline = ["log_moneyness", "T"]
        features_tree = ["log_moneyness", "T", "option_is_call"]

        X_tr_spline = self.df_train_full[features_spline].to_numpy(dtype=float)
        y_tr = self.df_train_full["iv_teorica"].to_numpy(dtype=float)
        X_tr_tree = self.df_train_full[features_tree].to_numpy(dtype=float)

        X_te_spline = self.df_test_puro[features_spline].to_numpy(dtype=float)
        X_te_tree = self.df_test_puro[features_tree].to_numpy(dtype=float)

        # 1. Spline baseline
        spline_pipe = Pipeline([
            ("tensor_spline", TensorBSplineFeatures(n_knots_m=5, n_knots_t=5, degree=3)),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 7))),
        ])
        spline_pipe.fit(X_tr_spline, y_tr)
        y_hat_spline_tr = spline_pipe.predict(X_tr_spline)
        y_hat_spline_te = spline_pipe.predict(X_te_spline)

        # 2. Residuals modeling
        residuals_tr = y_tr - y_hat_spline_tr
        xgb_resid = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.1, random_state=42)
        xgb_resid.fit(X_tr_tree, residuals_tr)
        e_hat_te = xgb_resid.predict(X_te_tree)

        # 3. Hybrid recombination
        y_hat_hybrid = y_hat_spline_te + e_hat_te

        self.assertEqual(len(y_hat_hybrid), len(self.df_test_puro))
        self.assertTrue(np.isfinite(y_hat_hybrid).all())
        print("  [OK] Hybrid Splines + XGBoost pipeline verified.")

    def test_10_surface_meshgrid_generation(self):
        """Test 3D surface and heatmap meshgrid creation."""
        print("Test 10: Testing 3D surface meshgrid & heatmap plotting...")
        df_sample = self.df_test_puro.copy()
        figs = af.plot_ivs_and_variance_surfaces(
            df_sample,
            x_col="log_moneyness",
            y_col="T",
            z_col="iv_teorica",
            nbins_x=20,
            nbins_y=15,
            show=False,
            return_figs=True,
        )
        self.assertIn("fig_iv_heatmap", figs)
        self.assertIn("fig_iv_surface", figs)
        self.assertIn("fig_w_heatmap", figs)
        self.assertIn("fig_w_surface", figs)
        plt.close("all")
        print("  [OK] 3D surface & heatmap generator verified.")

    def test_11_model_metadata_and_excel_integrity(self):
        """Verify that all 22 existing JSON metadata files and 12 Excel files load cleanly."""
        print("Test 11: Auditing all existing model metadata JSONs and Excel tables...")
        meta_dir = os.path.join(BASE_DIR, "04_outputs", "03_models_metadata")
        json_count = 0
        for root, _, files in os.walk(meta_dir):
            for f in files:
                if f.endswith(".json"):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    self.assertIsInstance(data, (dict, list), f"Invalid JSON in {fpath}")
                    json_count += 1
        self.assertGreaterEqual(json_count, 20, f"Expected >= 20 JSON metadata files, found {json_count}")
        print(f"  [OK] Validated {json_count} JSON metadata files.")

        tables_dir = os.path.join(BASE_DIR, "04_outputs", "02_tables")
        xlsx_count = 0
        for root, _, files in os.walk(tables_dir):
            for f in files:
                if f.endswith(".xlsx"):
                    fpath = os.path.join(root, f)
                    wb = openpyxl.load_workbook(fpath, read_only=True)
                    self.assertIn("globais", wb.sheetnames, f"Missing 'globais' sheet in {fpath}")
                    wb.close()
                    xlsx_count += 1
        self.assertGreaterEqual(xlsx_count, 12, f"Expected >= 12 Excel files, found {xlsx_count}")
        print(f"  [OK] Validated {xlsx_count} Excel files with 'globais' metric sheets.")


if __name__ == "__main__":
    unittest.main()
