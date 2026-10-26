"""
Auxiliary functions for SPX Implied Volatility Surface (IVS) modeling.
Exported from 01_auxiliary_functions.ipynb for direct Python module usage.
"""

# Color palette: blue, gray, black, and white tones
COLOR_AZUL1 = "#11469b"   # dark blue
COLOR_AZUL2 = "#2063c0"   # medium blue
COLOR_AZUL3 = "#5a90e7"   # light blue
COLOR_CINZA1 = "#777777"  # dark gray
COLOR_CINZA2 = "#AAAAAA"  # medium gray
COLOR_CINZA3 = "#bbbbbb"  # light gray
COLOR_PRETO = "#000000"   # black
COLOR_BRANCO = "#ffffff"  # white

# Named dictionary for convenient access
CORES = {
    'azul1': COLOR_AZUL1,
    'azul2': COLOR_AZUL2,
    'azul3': COLOR_AZUL3,
    'cinza1': COLOR_CINZA1,
    'cinza2': COLOR_CINZA2,
    'cinza3': COLOR_CINZA3,
    'preto': COLOR_PRETO,
    'branco': COLOR_BRANCO
}

PALETA_CORES = [COLOR_AZUL1, COLOR_AZUL2, COLOR_AZUL3, COLOR_CINZA1, COLOR_CINZA2, COLOR_CINZA3, COLOR_PRETO, COLOR_BRANCO]

# Named dictionary of line styles
ESTILOS_LINHA = {
    'contínua': '-',
    'tracejada': '--',
    'ponto-traço': '-.',
    'pontilhada': ':',
    'contínua2': '-',
    'tracejada2': '--',
    'ponto-traço2': '-.',
    'pontilhada2': ':'
}

# List of styles for compatibility (opcional)
ESTILOS_LINHA_LISTA = list(ESTILOS_LINHA.values())

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette(PALETA_CORES)

# Visualize color swatches
def mostrar_cores(paleta):
    fig, ax = plt.subplots(figsize=(8, 1))
    for i, cor in enumerate(paleta):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=cor))
        ax.text(i + 0.5, 0.5, cor, va='center', ha='center', fontsize=10, color='black' if cor not in ['#000000', '#404040'] else 'white')
    ax.set_xlim(0, len(paleta))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Palete de Cores')
    plt.show()

# Visualize line styles
def mostrar_estilos(estilos):
    fig, ax = plt.subplots(figsize=(8, 1))
    for i, (nome, estilo) in enumerate(estilos.items()):
        ax.plot([i, i+0.8], [0.5, 0.5], linestyle=estilo, color='black', linewidth=3)
        ax.text(i + 0.4, 0.7, nome, va='center', ha='center', fontsize=10, color='black')
    ax.set_xlim(0, len(estilos))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Estilos de Linha')
    plt.show()

# Display swatches in output
if __name__ == '__main__':
    print('Color Palette:')
    for nome, cor in CORES.items():
        print(f"{nome}: {cor}")
    
    print('\nLine Styles:')
    for nome, estilo in ESTILOS_LINHA.items():
        print(f"{nome}: {estilo}")
    
    # show_colors(COLOR_PALETTE)
    # show_line_styles(LINE_STYLES)
# English compatibility aliases
COLORS = CORES
COLOR_PALETTE = PALETA_CORES
LINE_STYLES = ESTILOS_LINHA
show_colors = mostrar_cores
show_line_styles = mostrar_estilos


def plot_ivs_and_variance_surfaces(
    df,
    x_col="log_moneyness",     # X axis
    y_col="T",                 # Y axis (years)
    z_col="iv_teorica",        # IV in annualized decimal
    agg="median",              # "median" ou "mean"
    nbins_x=50,
    nbins_y=40,
    qtrim_x=(0.01, 0.99),
    qtrim_y=(0.01, 0.99),
    min_count_per_bin=5,       # filter bins with low observation count
    # Optional scales — ensures visual consistency across models
    iv_vmin=None, iv_vmax=None,    # IV in % (visual display only)
    w_vmin=None,  w_vmax=None,     # Total variance w in decimal
    # Output file export paths
    save_dir=None,
    filename_prefix="ivs",
    # Display and return
    show=True,                 # se False, não dá plt.show() (útil para pipelines)
    return_figs=False          # se True, retorna dict com as figuras (sem grids)
):
    """
    Generates 4 figures:
      1) Heatmap IV (%)
      2) 3D IV Surface (%)
      3) Heatmap Variância Total w
      4) 3D Total Variance Surface w

    Does not return grids/arrays. Optionally returns only figure objects.
    """
    import numpy as np, pandas as pd, os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _build_grid(df_in, x_col, y_col, z_series, agg, nbx, nby, qx, qy, mincnt):
        d = df_in[[x_col, y_col]].copy()
        d["_z"] = z_series.values
        d = d.dropna()
        if d.empty:
            raise ValueError("Sem dados válidos para visualizar.")

        # trims
        qx0, qx1 = d[x_col].quantile(list(qx))
        qy0, qy1 = d[y_col].quantile(list(qy))
        d = d[d[x_col].between(qx0, qx1) & d[y_col].between(max(0.0, qy0), qy1)]
        if d.empty:
            raise ValueError("Sem dados após trims.")

        # bins
        x_bins = np.linspace(float(qx0), float(qx1), nbx + 1)
        y_bins = np.linspace(float(max(0.0, qy0)), float(qy1), nby + 1)
        d["_xb"] = pd.cut(d[x_col], bins=x_bins, labels=False, include_lowest=True)
        d["_yb"] = pd.cut(d[y_col], bins=y_bins, labels=False, include_lowest=True)

        # aggregation
        if agg == "mean":
            grid = d.groupby(["_yb", "_xb"])["_z"].mean().unstack("_xb")
            counts = d.groupby(["_yb", "_xb"])["_z"].size().unstack("_xb")
        else:
            grid = d.groupby(["_yb", "_xb"])["_z"].median().unstack("_xb")
            counts = d.groupby(["_yb", "_xb"])["_z"].size().unstack("_xb")

        # filter bins with low observation count
        grid = grid.where(counts >= mincnt)

        # drop empty cells
        grid = grid.dropna(how="all").dropna(axis=1, how="all")
        if grid.size == 0:
            raise ValueError("Grelha ficou vazia após agregação/filtragem.")

        # true axis values
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        xi = grid.columns.to_numpy(dtype=int)
        yi = grid.index.to_numpy(dtype=int)
        X_vals = x_centers[xi]
        Y_vals = y_centers[yi]
        Z = grid.values.copy()

        return X_vals, Y_vals, Z

    # --- prepare IV (%) and total variance w ---
    _df = df[[x_col, y_col, z_col]].dropna().copy()
    if _df.empty:
        raise ValueError("Sem dados para visualização.")

    iv_dec = _df[z_col].astype(float).clip(lower=0)  # robustez mínima
    w = (iv_dec ** 2) * _df[y_col].astype(float)

    # Grids (sem retornar)
    X_iv, Y_iv, Z_iv_dec = _build_grid(_df, x_col, y_col, iv_dec, agg, nbins_x, nbins_y, qtrim_x, qtrim_y, min_count_per_bin)
    X_w,  Y_w,  Z_w      = _build_grid(_df, x_col, y_col, w,      agg, nbins_x, nbins_y, qtrim_x, qtrim_y, min_count_per_bin)

    # Convert IV to % for plotting
    Z_iv_pct = Z_iv_dec * 100.0

    # --- plotting helpers ---
    def _heatmap(Z, Xvals, Yvals, label, title, vmin=None, vmax=None):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        sns.heatmap(
            Z, cmap="viridis",
            xticklabels=[f"{x:.2f}" for x in Xvals],
            yticklabels=[f"{y:.2f}" for y in Yvals],
            cbar_kws={"label": label},
            vmin=vmin, vmax=vmax, ax=ax
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"{y_col} (Years)")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def _surface3d(Z, Xvals, Yvals, zlabel, title):
        X, Y = np.meshgrid(Xvals, Yvals)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"{y_col} (Years)")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        cb = fig.colorbar(surf, shrink=0.6, aspect=18)
        cb.set_label(zlabel)
        fig.tight_layout()
        return fig

    # --- 4 plots: IV surface, w surface, IV heatmap, w heatmap ---
    fig_iv_hm = _heatmap(Z_iv_pct, X_iv, Y_iv, "IV (%)",
                         f"Implied Volatility Surface — Heatmap ({agg})", vmin=iv_vmin, vmax=iv_vmax)
    fig_iv_3d = _surface3d(Z_iv_pct, X_iv, Y_iv, "IV (%)",
                           f"Implied Volatility Surface — 3D ({agg})")

    fig_w_hm  = _heatmap(Z_w, X_w, Y_w, "Total Variance (w)",
                         f"Total Variance Surface — Heatmap ({agg})", vmin=w_vmin, vmax=w_vmax)
    fig_w_3d  = _surface3d(Z_w, X_w, Y_w, "Total Variance (w)",
                           f"Total Variance Surface — 3D ({agg})")

    # Save figures
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_iv_hm.savefig(os.path.join(save_dir, f"{filename_prefix}_iv_heatmap.png"), dpi=150, bbox_inches="tight")
        fig_iv_3d.savefig(os.path.join(save_dir, f"{filename_prefix}_iv_surface.png"), dpi=150, bbox_inches="tight")
        fig_w_hm.savefig( os.path.join(save_dir, f"{filename_prefix}_w_heatmap.png"),  dpi=150, bbox_inches="tight")
        fig_w_3d.savefig( os.path.join(save_dir, f"{filename_prefix}_w_surface.png"),  dpi=150, bbox_inches="tight")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        # headless execution (no plt.show)
        pass

    if return_figs:
        return {
            "fig_iv_heatmap": fig_iv_hm,
            "fig_iv_surface": fig_iv_3d,
            "fig_w_heatmap":  fig_w_hm,
            "fig_w_surface":  fig_w_3d,
        }
    # por defeito, não retorna nada → não “sujamos” o output


def run_avaliacoes(
    df_inner_train,
    df_inner_tunning,
    df_train_full,
    df_test_puro,
    *,
    model: str,                           # "OLS", "RF", "XGB", "Spline", ...
    DIR_TABELAS: str,
    DIR_GRAFICOS: str,
    excel_name: str | None = None,        # default = f"{model.lower()}_metricas.xlsx"
    min_count_bin: int = 20,              # mínimo de observações para validar cada célula 5×5
    n_bins_line: int = 20,                # bins (quantis) para a linha média nos gráficos vs m/T
    dpi: int = 160,
    scatter_mode: str = "all",            # "all" | "subsample" | "hexbin"
    max_points_scatter: int | None = None,# only if scatter_mode="subsample"
    random_state: int = 42,
    display_in_notebook: bool = True      # mostra tabela globais e plt.show() dos gráficos
) -> dict:
    """
    Executa 4 runs fixos (1 geral + 3 específicas), gera PNGs e atualiza um Excel único por modelo:
      - 'globais' com métricas por run
      - Sheets 5x5 por métrica (apenas para a run geral de 'test_puro'):
        'Analise BIN | RMSE', '... | MAE', '... | MedAE', '... | Bias', '... | Count'

    Requisitos por DataFrame: colunas ['iv_teorica','ivs_modelo','log_moneyness','T'] sem NaN/Inf.
    Palette colors used: COLOR_AZUL1/2/3, COLOR_CINZA1/2/3, COLOR_PRETO, COLOR_BRANCO.
    """
    # ----------------- imports locais -----------------
    import os, math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # ----------------- compat / normalização -----------------
    REQUIRED_COLS = ["iv_teorica", "ivs_modelo", "log_moneyness", "T"]

    def _ensure_pred_col(df: pd.DataFrame) -> pd.DataFrame:
        """Garante a existência de 'ivs_modelo'. Se existir 'ivs_modelo', cria uma cópia e mapeia."""
        if "ivs_modelo" not in df.columns:
            for legacy in ("ivs_modelo",):
                if legacy in df.columns:
                    d = df.copy()
                    d["ivs_modelo"] = d[legacy]
                    return d
        return df

    def _fail_fast(df, split):
        miss = [c for c in REQUIRED_COLS if c not in df.columns]
        if miss:
            raise ValueError(f"[{split}] Missing columns: {miss}")
        arr = df[REQUIRED_COLS].to_numpy()
        if not np.isfinite(arr).all():
            raise ValueError(f"[{split}] Existem NaN/Inf nas colunas obrigatórias.")
        if len(df) == 0:
            raise ValueError(f"[{split}] DataFrame vazio.")

    def _rmse(e):   return float(np.sqrt(np.mean(e**2)))
    def _mae(e):    return float(np.mean(np.abs(e)))
    def _medae(e):  return float(np.median(np.abs(e)))
    def _bias(e):   return float(np.mean(e))

    def _r2(y_true, y_pred):
        var = np.var(y_true)
        if var == 0:
            return float("nan")
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def _binned_line(x, y, n_bins=n_bins_line):
        """
        Divide o eixo-x em quantis (n_bins) e devolve:
          xc: média de x por bin
          yc: média de y por bin (linha de tendência)
          ys: desvio-padrão de y por bin (para banda ±1σ)
        """
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        for k in range(1, len(edges)):
            if edges[k] <= edges[k-1]:
                edges[k] = edges[k-1] + 1e-12
        idx = np.digitize(x, edges[1:-1], right=True)
        xc, yc, ys = [], [], []
        for b in range(n_bins):
            mask = (idx == b)
            if mask.any():
                xc.append(np.mean(x[mask]))
                yc.append(np.mean(y[mask]))
                ys.append(np.std(y[mask]))
        return np.array(xc), np.array(yc), np.array(ys)

    def _maybe_subsample(df):
        if scatter_mode == "subsample" and max_points_scatter and len(df) > max_points_scatter:
            return df.sample(max_points_scatter, random_state=random_state)
        return df

    # ----------------- colormap para hexbin (cores da paleta) -----------------
    hexbin_cmap = LinearSegmentedColormap.from_list(
        "hexbin_map", [COLOR_BRANCO, COLOR_AZUL2, COLOR_AZUL3]
    )

    # ----------------- Visualizations with Color Palette -----------------
    def _plot_scatter(df, save_path, title=None, show=False):
        d = _maybe_subsample(df)
        plt.figure(figsize=(6.5, 6.5))
        if scatter_mode == "hexbin":
            plt.hexbin(d["iv_teorica"], d["ivs_modelo"], gridsize=60, cmap=hexbin_cmap)
        else:
            plt.scatter(d["iv_teorica"], d["ivs_modelo"], s=6, alpha=0.35, edgecolor="none", color=COLOR_AZUL2)
        # 45-degree line
        mn = float(np.nanmin([d["iv_teorica"].min(), d["ivs_modelo"].min()]))
        mx = float(np.nanmax([d["iv_teorica"].max(), d["ivs_modelo"].max()]))
        plt.plot([mn, mx], [mn, mx], linewidth=1.1, color=COLOR_PRETO)
        plt.xlabel("Observed IV")
        plt.ylabel(f"Predicted IV ({model})")
        if title: plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=COLOR_CINZA3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    def _plot_hist_residuos(df, save_path, title=None, bins="fd", show=False):
        resid = (df["ivs_modelo"] - df["iv_teorica"]).to_numpy()
        plt.figure(figsize=(7.2, 5.0))
        plt.hist(resid, bins=bins, alpha=0.9, color=COLOR_AZUL1)
        plt.axvline(0.0, linewidth=1.0, color=COLOR_PRETO)
        plt.xlabel(f"Residual (IV_{model.lower()} - IV_observed)")
        plt.ylabel("Frequency")
        if title: plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=COLOR_CINZA3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    def _plot_resid_vs_m(df, save_path, title=None, show=False):
        x = df["log_moneyness"].to_numpy()
        y = (df["ivs_modelo"] - df["iv_teorica"]).to_numpy()
        plt.figure(figsize=(7.8, 5.0))
        if scatter_mode == "hexbin":
            plt.hexbin(x, y, gridsize=60, cmap=hexbin_cmap)
        else:
            d = _maybe_subsample(df)
            plt.scatter(d["log_moneyness"], (d["ivs_modelo"] - d["iv_teorica"]),
                        s=4, alpha=0.15, edgecolor="none", color=COLOR_AZUL1)
        xc, yc, ys = _binned_line(x, y, n_bins=n_bins_line)
        if len(xc) > 0:
            plt.plot(xc, yc, linewidth=1.6, color=COLOR_AZUL3)
            plt.fill_between(xc, yc - ys, yc + ys, alpha=0.12, color=COLOR_AZUL3)
        plt.axhline(0.0, linewidth=1.0, color=COLOR_PRETO)
        plt.xlabel("Log-moneyness (m)")
        plt.ylabel(f"Residual (IV_{model.lower()} - IV_observed)")
        if title: plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=COLOR_CINZA3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    def _plot_resid_vs_T(df, save_path, title=None, show=False):
        x = df["T"].to_numpy()
        y = (df["ivs_modelo"] - df["iv_teorica"]).to_numpy()
        plt.figure(figsize=(7.8, 5.0))
        if scatter_mode == "hexbin":
            plt.hexbin(x, y, gridsize=60, cmap=hexbin_cmap)
        else:
            d = _maybe_subsample(df)
            plt.scatter(d["T"], (d["ivs_modelo"] - d["iv_teorica"]),
                        s=4, alpha=0.15, edgecolor="none", color=COLOR_AZUL1)
        xc, yc, ys = _binned_line(x, y, n_bins=n_bins_line)
        if len(xc) > 0:
            plt.plot(xc, yc, linewidth=1.6, color=COLOR_AZUL3)
            plt.fill_between(xc, yc - ys, yc + ys, alpha=0.12, color=COLOR_AZUL3)
        plt.axhline(0.0, linewidth=1.0, color=COLOR_PRETO)
        plt.xlabel("Maturity T (years)")
        plt.ylabel(f"Residual (IV_{model.lower()} - IV_observed)")
        if title: plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=COLOR_CINZA3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    # ----------------- bins 5×5 em matriz -----------------
    def _compute_5x5_mats(df):
        """Devolve dicionário com 5 matrizes (RMSE, MAE, MedAE, Bias, Count) 5×5."""
        y_true = df["iv_teorica"].to_numpy()
        y_pred = df["ivs_modelo"].to_numpy()
        e = y_pred - y_true
        m = df["log_moneyness"].to_numpy()
        T = df["T"].to_numpy()

        m_edges = np.quantile(m, [0, .2, .4, .6, .8, 1.0])
        T_edges = np.quantile(T, [0, .2, .4, .6, .8, 1.0])
        for v in (m_edges, T_edges):
            for i in range(1, len(v)):
                if v[i] <= v[i-1]:
                    v[i] = v[i-1] + 1e-12

        m_bin = np.digitize(m, m_edges[1:-1], right=True) + 1  # 1..5 (colunas)
        T_bin = np.digitize(T, T_edges[1:-1], right=True) + 1  # 1..5 (linhas)

        RMSE = np.full((5,5), np.nan)
        MAE  = np.full((5,5), np.nan)
        MedAE= np.full((5,5), np.nan)
        Bias = np.full((5,5), np.nan)
        Cnt  = np.zeros((5,5), dtype=int)

        for i in range(1,6):        # m_bin (coluna)
            for j in range(1,6):    # T_bin (linha)
                mask = (m_bin == i) & (T_bin == j)
                cnt = int(mask.sum())
                Cnt[j-1, i-1] = cnt  # linhas = T_bin, colunas = m_bin
                if cnt >= min_count_bin:
                    ee = e[mask]
                    RMSE[j-1, i-1] = _rmse(ee)
                    MAE[j-1,  i-1] = _mae(ee)
                    MedAE[j-1,i-1] = _medae(ee)
                    Bias[j-1, i-1] = _bias(ee)

        return {"RMSE": RMSE, "MAE": MAE, "MedAE": MedAE, "Bias": Bias, "Count": Cnt}

    def _write_matrix_sheet(writer, sheet_name, M):
        # labels: columns m1..m5, rows T1..T5
        dfM = pd.DataFrame(M, index=[f"T{i}" for i in range(1,6)], columns=[f"m{i}" for i in range(1,6)])
        dfM.index.name = "T_bin"
        dfM.columns.name = "m_bin"
        dfM.to_excel(writer, sheet_name=sheet_name)

    # ----------------- Excel: leitura/escrita -----------------
    def _read_existing_globais(excel_path):
        if os.path.exists(excel_path):
            try:
                return pd.read_excel(excel_path, sheet_name="globais")
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _write_excel(excel_path, globais_df, mats_5x5=None):
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as wr:
            globais_df.to_excel(wr, sheet_name="globais", index=False)
            if mats_5x5 is not None:
                _write_matrix_sheet(wr, "Bin Analysis | RMSE", mats_5x5["RMSE"])  # translated
                _write_matrix_sheet(wr, "Bin Analysis | MAE",  mats_5x5["MAE"])   # translated
                _write_matrix_sheet(wr, "Bin Analysis | MedAE",mats_5x5["MedAE"]) # translated
                _write_matrix_sheet(wr, "Bin Analysis | Bias", mats_5x5["Bias"])  # translated
                _write_matrix_sheet(wr, "Bin Analysis | Count",mats_5x5["Count"]) # translated

    # ----------------- preparar runs (fixos) -----------------
    runs = [
        {"split": "Test Sample",     "df": df_test_puro,     "analysis": "geral",      "variant": "train_full"},
        {"split": "Train Sample",    "df": df_train_full,    "analysis": "especifica", "variant": "train_full"},
        {"split": "inner_train",   "df": df_inner_train,   "analysis": "especifica", "variant": "inner_training"},
        {"split": "inner_tunning", "df": df_inner_tunning, "analysis": "especifica", "variant": "inner_training"},
    ]

    # ----------------- loop principal -----------------
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    os.makedirs(DIR_TABELAS, exist_ok=True)

    excel_path = os.path.join(DIR_TABELAS, (excel_name or f"{model.lower()}_metricas.xlsx"))
    old_globais = _read_existing_globais(excel_path)

    new_rows = []
    mats_5x5 = None  # só para a run geral

    for r in runs:
        split    = r["split"]
        df_in    = r["df"]
        analysis = r["analysis"]
        variant  = r["variant"]

        # compat: garantir 'ivs_modelo'
        df = _ensure_pred_col(df_in)

        _fail_fast(df, split)

        y_true = df["iv_teorica"].to_numpy()
        y_pred = df["ivs_modelo"].to_numpy()
        e      = y_pred - y_true

        # compute global error metrics (RMSE, MAE, MedAE, R2, Bias)
        row = {
            "model": model,
            "split": split,
            "analysis": analysis,
            "variant": variant,
            "n_obs": int(len(df)),
            "RMSE": _rmse(e),
            "MAE": _mae(e),
            "MedAE": _medae(e),
            "R2": _r2(y_true, y_pred),
            "Bias": _bias(e),
        }
        new_rows.append(row)

        # gráficos — também mostrar (plt.show) se display_in_notebook=True
        base = model.lower()
        show = bool(display_in_notebook)
        if analysis == "geral":
            _plot_scatter    (df, os.path.join(DIR_GRAFICOS, f"{base}_scatter_{split}.png"),
                              title=f"Observed vs Predicted IV — {model} ({split})", show=show)
            _plot_hist_residuos(df, os.path.join(DIR_GRAFICOS, f"{base}_resid_hist_{split}.png"),
                                title=f"Residuals Histogram (IV_{model.lower()} - IV_observed) — {model} ({split})", show=show)
            _plot_resid_vs_m  (df, os.path.join(DIR_GRAFICOS, f"{base}_resid_vs_m_{split}.png"),
                                title=f"Residuals vs m — {model} ({split})", show=show)
            _plot_resid_vs_T  (df, os.path.join(DIR_GRAFICOS, f"{base}_resid_vs_T_{split}.png"),
                                title=f"Residuals vs T — {model} ({split})", show=show)
            mats_5x5 = _compute_5x5_mats(df)
        else:
            _plot_scatter(df, os.path.join(DIR_GRAFICOS, f"{base}_scatter_{split}.png"),
                          title=f"Observed vs Predicted IV — {model} ({split})", show=show)

    # consolidar globais (substitui linhas destes 4 splits do mesmo modelo; preserva outras entradas)
    globais_df_new = pd.DataFrame(new_rows)
    if not old_globais.empty:
        mask_same_model = (old_globais["model"] == model) & (old_globais["split"].isin(globais_df_new["split"].unique()))
        old_globais = old_globais[~mask_same_model]
        globais_out = pd.concat([old_globais, globais_df_new], ignore_index=True)
    else:
        globais_out = globais_df_new

    # write to Excel workbook
    _write_excel(excel_path, globais_out, mats_5x5=mats_5x5)

    # display global metrics summary
    if display_in_notebook:
        try:
            from IPython.display import display  # type: ignore
            display(globais_out)
        except Exception:
            print("\nTabela 'globais':")
            print(globais_out.to_string(index=False))

    return {
        "excel_path": excel_path,
        "globais_rows": int(len(globais_out)),
        "has_bins_5x5": mats_5x5 is not None,
        "graficos_dir": DIR_GRAFICOS,
        "globais_df": globais_out,   # devolve a tabela para inspeção/uso no notebook
    }
# English compatibility alias
run_evaluations = run_avaliacoes


