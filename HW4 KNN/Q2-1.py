"""Q2-1：单细胞类型注释，Emerge式集成KNN与特征/集成消融。

依赖：numpy pandas scipy scikit-learn matplotlib umap-learn threadpoolctl
检查：python Q2-1.py --check-only
验证：python Q2-1.py --profile smoke
正式：python Q2-1.py
重绘：python Q2-1.py --redraw-from Q2-1_smoke_results --output-dir Q2-1_polished
图谱：python Q2-1.py --atlas-only

默认读取脚本同目录 data2/，输出 Q2-1_results/。
正式模式使用全部细胞、3个种子、A-D特征、full/matched来源对照。
smoke显式分层抽样180细胞/来源，仅用于流程验证，不支持正式性能结论。
五种距离使用精确近邻；训练距离尺度固定，测试标签仅用于评价。
共有基因按gene_name对齐，同名列求和；不依赖253基因映射或MERFISH。
UMAP和marker图使用真实测量，只用于展示，不构造组织坐标。
探索性联合UMAP默认另取1500细胞/来源，不参与分类；所有图白底，无workflow图。
代码依据：Emerge Supplementary Note 3.2；ENGEP多配置集成与比较图。
"""
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse


def _npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if dtype.hasobject or fortran:
        raise ValueError("Object or Fortran-order sparse storage is unsupported.")
    return tuple(shape), dtype


def inspect_dataset(data_dir, name, label_column="subclass_label"):
    """Read CSV metadata and NPZ headers, preserving their original row order.

    This checks metadata/shape/storage correspondence, not every matrix value.
    load_expression performs the complete streamed numeric/index validation.
    Missing required files or labels are errors, never synthesized or skipped.
    """
    directory = Path(data_dir) / name
    matrix_name = "X_exon.npz" if name.startswith("SMARTer") else "X_counts.npz"
    paths = {key: directory / value for key, value in {
        "cells": "cells.csv", "genes": "genes.csv", "npz": matrix_name
    }.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    cells = pd.read_csv(paths["cells"], dtype=str, keep_default_na=False)
    genes = pd.read_csv(paths["genes"], dtype=str, keep_default_na=False)
    for column in ("cell_id", label_column):
        if column not in cells:
            raise ValueError(f"{name}: cells.csv lacks {column!r}.")
        cells[column] = cells[column].str.strip()
        if cells[column].eq("").any():
            raise ValueError(f"{name}: empty values in cells.csv/{column}.")
    if cells["cell_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate cell_id values.")
    if "gene_name" not in genes:
        raise ValueError(f"{name}: genes.csv lacks gene_name.")
    genes["gene_name"] = genes["gene_name"].str.strip()
    with zipfile.ZipFile(paths["npz"]) as archive:
        required = {f"{key}.npy" for key in ("shape", "format", "data", "indices", "indptr")}
        if not required.issubset(archive.namelist()):
            raise ValueError(f"{name}: incomplete scipy sparse NPZ archive.")
        with archive.open("shape.npy") as stream:
            shape_array = np.load(stream, allow_pickle=False)
        with archive.open("format.npy") as stream:
            matrix_format = np.load(stream, allow_pickle=False).item()
        if isinstance(matrix_format, bytes):
            matrix_format = matrix_format.decode("ascii")
        if matrix_format not in {"csr", "csc"}:
            raise ValueError(f"{name}: expected CSR or CSC, got {matrix_format!r}.")
        if shape_array.shape != (2,) or shape_array.dtype.kind not in "iu":
            raise ValueError(f"{name}: invalid matrix shape array.")
        shape = tuple(int(value) for value in shape_array)
        if shape != (len(cells), len(genes)) or min(shape) <= 0:
            raise ValueError(f"{name}: matrix {shape} disagrees with CSV rows "
                             f"({len(cells)}, {len(genes)}); no automatic transpose.")
        headers = {}
        for key in ("data", "indices", "indptr"):
            with archive.open(f"{key}.npy") as stream:
                headers[key] = _npy_header(stream)
        nnz_shape, data_dtype = headers["data"]
        if len(nnz_shape) != 1 or data_dtype.kind not in "iuf":
            raise ValueError(f"{name}: invalid numeric data storage.")
        if headers["indices"][0] != nnz_shape or headers["indices"][1].kind not in "iu":
            raise ValueError(f"{name}: invalid sparse indices storage.")
        outer_size = shape[0] if matrix_format == "csr" else shape[1]
        if headers["indptr"][0] != (outer_size + 1,) or headers["indptr"][1].kind not in "iu":
            raise ValueError(f"{name}: invalid sparse indptr storage.")
    return {"name": name, "cells": cells, "genes": genes,
            "label_column": label_column, "npz_path": paths["npz"],
            "matrix_shape": shape, "format": matrix_format,
            "nnz": nnz_shape[0], "headers": headers}


def common_gene_names(meta_list):
    """All nonempty gene symbols shared by every reference; not a 253-gene panel."""
    if not meta_list:
        raise ValueError("At least one dataset is required.")
    shared = None
    for meta in meta_list:
        symbols = set(meta["genes"]["gene_name"]) - {""}
        shared = symbols if shared is None else shared.intersection(symbols)
    if not shared:
        raise ValueError("The reference datasets have no shared nonempty gene_name.")
    return np.asarray(sorted(shared), dtype=str)


def _npz_blocks(archive, key, expected_header, block_entries=524288):
    with archive.open(f"{key}.npy") as stream:
        shape, dtype = _npy_header(stream)
        if (shape, dtype) != expected_header:
            raise ValueError(f"NPZ {key} header changed since inspection.")
        for start in range(0, shape[0], block_entries):
            count = min(block_entries, shape[0] - start)
            raw = stream.read(count * dtype.itemsize)
            if len(raw) != count * dtype.itemsize:
                raise ValueError(f"Truncated NPZ array: {key}")
            yield start, np.frombuffer(raw, dtype=dtype)
        if stream.read(1):
            raise ValueError(f"Unexpected trailing bytes in NPZ array: {key}")


def _mapped_coordinates(start, indices, indptr, matrix_format, row_map, col_map):
    # Recover the compressed row (CSR) or column (CSC) for each entry.
    outer = np.searchsorted(indptr, np.arange(start, start + len(indices)), side="right") - 1
    if matrix_format == "csr":
        rows, columns = row_map[outer], col_map[indices]
    else:
        rows, columns = row_map[indices], col_map[outer]
    keep = (rows >= 0) & (columns >= 0)
    return rows[keep], columns[keep], keep


def load_expression(meta, rows, common_genes, cache_dir=None):
    """Return float32 CSR in requested row/gene order; sum duplicate symbols.

    rows=None selects all cells. Input row indices must be unique. Every stored
    value and index is validated even when only a small row subset is requested.
    cache_dir is accepted for API compatibility; this function writes no cache.
    A small subset saves RAM, but compressed NPZ still needs a full sequential
    scan. Full mode allocates the retained sparse matrix, never a dense matrix.
    """
    del cache_dir
    n_cells, n_genes = meta["matrix_shape"]
    rows = np.arange(n_cells, dtype=np.int64) if rows is None else np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise ValueError("rows must be a nonempty one-dimensional integer array.")
    if np.any(rows < 0) or np.any(rows >= n_cells) or len(np.unique(rows)) != len(rows):
        raise ValueError("rows must contain unique, in-range cell indices.")
    common_genes = np.asarray(common_genes, dtype=str)
    if common_genes.ndim != 1 or not len(common_genes) or np.any(common_genes == ""):
        raise ValueError("common_genes must contain nonempty gene symbols.")
    if len(np.unique(common_genes)) != len(common_genes):
        raise ValueError("common_genes must not contain duplicates.")
    symbol_to_col = {symbol: i for i, symbol in enumerate(common_genes)}
    col_map = np.asarray([symbol_to_col.get(symbol, -1)
                          for symbol in meta["genes"]["gene_name"]], dtype=np.int64)
    if len(np.unique(col_map[col_map >= 0])) != len(common_genes):
        raise ValueError(f"{meta['name']}: requested gene symbols are missing.")
    row_map = np.full(n_cells, -1, dtype=np.int64)
    row_map[rows] = np.arange(len(rows))
    counts = np.zeros(len(rows), dtype=np.int64)
    with zipfile.ZipFile(meta["npz_path"]) as archive:
        with archive.open("indptr.npy") as stream:
            indptr = np.load(stream, allow_pickle=False)
        expected_outer = n_cells if meta["format"] == "csr" else n_genes
        if (indptr.ndim != 1 or len(indptr) != expected_outer + 1
                or indptr.dtype.kind not in "iu" or indptr[0] != 0
                or indptr[-1] != meta["nnz"] or np.any(indptr[1:] < indptr[:-1])):
            raise ValueError(f"{meta['name']}: invalid sparse pointer array.")
        inner_bound = n_genes if meta["format"] == "csr" else n_cells
        for start, indices in _npz_blocks(archive, "indices", meta["headers"]["indices"]):
            if np.any(indices < 0) or np.any(indices >= inner_bound):
                raise ValueError(f"{meta['name']}: sparse index outside matrix shape.")
            out_rows, _, _ = _mapped_coordinates(start, indices, indptr, meta["format"], row_map, col_map)
            counts += np.bincount(out_rows, minlength=len(rows))
        output_ptr = np.r_[0, np.cumsum(counts)]
        output_nnz = int(output_ptr[-1])
        index_dtype = np.int32 if max(output_nnz, len(common_genes)) < 2**31 else np.int64
        output_ptr = output_ptr.astype(index_dtype, copy=False)
        output_indices = np.empty(output_nnz, dtype=index_dtype)
        output_data = np.empty(output_nnz, dtype=np.float32)
        cursor = output_ptr[:-1].astype(np.int64, copy=True)
        index_blocks = _npz_blocks(archive, "indices", meta["headers"]["indices"])
        data_blocks = _npz_blocks(archive, "data", meta["headers"]["data"])
        for (start, indices), (data_start, values) in zip(index_blocks, data_blocks):
            if start != data_start or len(indices) != len(values):
                raise ValueError("Sparse data/indices streams are misaligned.")
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{meta['name']}: expression contains nonfinite or negative values.")
            out_rows, out_cols, keep = _mapped_coordinates(start, indices, indptr, meta["format"], row_map, col_map)
            if not len(out_rows):
                continue
            # Stable row grouping fills CSR directly, avoiding a full COO copy.
            order = np.argsort(out_rows, kind="stable")
            grouped_rows = out_rows[order]
            unique_rows, group_start, group_count = np.unique(
                grouped_rows, return_index=True, return_counts=True)
            positions = (np.repeat(cursor[unique_rows], group_count)
                         + np.arange(len(order)) - np.repeat(group_start, group_count))
            output_indices[positions] = out_cols[order]
            output_data[positions] = values[keep][order]
            cursor[unique_rows] += group_count
    if not np.array_equal(cursor, output_ptr[1:]):
        raise ValueError("Incomplete sparse output; input may have changed while reading.")
    result = sparse.csr_matrix((output_data, output_indices, output_ptr),
                               shape=(len(rows), len(common_genes)), copy=False)
    result.sum_duplicates()  # Duplicate gene_name columns are summed, not discarded.
    result.eliminate_zeros()
    result.sort_indices()
    for start in range(0, result.nnz, 524288):
        if not np.isfinite(result.data[start:start + 524288]).all():
            raise ValueError("Expression overflows float32 after duplicate-gene summation.")
    return result


from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator

DATASET_SHORT = {'10X_cells_v2_AIBS': '10X v2', '10X_cells_v3_AIBS': '10X v3', 'SMARTer_cells_MOp': 'SMARTer'}
DATASET_COLORS = {'10X_cells_v2_AIBS': '#397CA8', '10X_cells_v3_AIBS': '#D08047', 'SMARTer_cells_MOp': '#7764A6'}
MODEL_ORDER = ['single', 'multi_k', 'ensemble']
MODEL_NAMES = {'single': 'Single kNN', 'multi_k': 'Multi-k', 'ensemble': 'Ensemble'}
MODEL_COLORS = {'single': '#8D9EAC', 'multi_k': '#3E8EAA', 'ensemble': '#BB6177'}
FEATURE_ORDER = ['A_raw', 'B_log', 'C_hvg2000', 'D_hvg4000']
FEATURE_NAMES = dict(zip(FEATURE_ORDER, ['Raw', 'Log', '2k HVGs', '4k HVGs']))
METRIC_NAMES = {'accuracy': 'Accuracy', 'macro_f1': 'Macro F1', 'balanced_accuracy': 'Balanced accuracy'}
LABEL_ORDER = ['L2/3 IT', 'L5 IT', 'L6 IT', 'L6 IT Car3', 'L5 ET', 'L5/6 NP', 'L6 CT', 'L6b',
               'Lamp5', 'Sncg', 'Vip', 'Sst', 'Pvalb', 'Astro', 'OPC', 'Oligo', 'Endo', 'VLMC', 'SMC', 'Macrophage']
_TYPE_COLORS = ['#D9843D', '#D75545', '#AF3F52', '#A47246', '#E4AF39', '#CD7866', '#8E5266', '#A89562',
                '#6DAA57', '#A1B552', '#3A9C92', '#447A49', '#315E72', '#547FA9', '#8197CD', '#6662A6',
                '#9B73A6', '#C083AA', '#7F8996', '#AB7E69']


def _style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.titlesize': 18,
        'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'legend.fontsize': 14, 'text.color': '#263643', 'axes.labelcolor': '#263643',
        'xtick.color': '#43515D', 'ytick.color': '#43515D', 'axes.edgecolor': '#8B969E',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'savefig.transparent': False, 'figure.dpi': 110, 'savefig.dpi': 220,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{name}: missing required columns {sorted(missing)}')
    if frame.empty:
        raise ValueError(f'{name}: no observations available')


def _select(frame, name='results', **filters):
    result = frame.copy()
    for column, value in filters.items():
        if value is not None:
            _require(result, [column], name)
            result = result.loc[result[column].eq(value)]
    if result.empty:
        raise ValueError(f'{name}: no rows match {filters}')
    return result


def _slug(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('_') or 'dataset'


def _seed_note(frame):
    n = frame['seed'].nunique()
    return f'Mean ± SD · {n} seeds' if n > 1 else 'One seed'


def _save(fig, out_dir, name, save_pdf=False, note=None):
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
    note = globals().get('FIGURE_NOTE', '') if note is None else note
    if note:
        fig.text(0.5, -0.015, note, ha='center', va='top', fontsize=14, color='#60717E')
    path = folder / f'{name}.png'
    fig.savefig(path, bbox_inches='tight', facecolor='white', transparent=False, pad_inches=0.22)
    if save_pdf:
        fig.savefig(folder / f'{name}.pdf', bbox_inches='tight', facecolor='white', transparent=False, pad_inches=0.22)
    plt.close(fig)
    return path


def _ordered_labels(labels):
    available = {str(label) for label in labels}
    return [label for label in LABEL_ORDER if label in available] + sorted(available - set(LABEL_ORDER))


def make_label_palette(labels):
    palette = dict(zip(LABEL_ORDER, _TYPE_COLORS))
    extra = sorted({str(label) for label in labels} - set(palette))
    for index, label in enumerate(extra):
        palette[label] = plt.get_cmap('tab20')(index % 20)
    return {label: palette[label] for label in _ordered_labels(labels)}


def _palette(labels, palette):
    labels = {str(label) for label in labels}
    if palette is None:
        return make_label_palette(labels)
    missing = labels - set(palette)
    if missing:
        raise ValueError(f'Palette lacks labels: {sorted(missing)}')
    return palette


def _diagnostics(diagnostics):
    if isinstance(diagnostics, dict):
        if not diagnostics:
            raise ValueError('diagnostics: no datasets supplied')
        data = pd.concat([frame.assign(dataset=name) for name, frame in diagnostics.items()], ignore_index=True)
    else:
        data = diagnostics.copy()
    _require(data, ['dataset', 'true_label', 'predicted_label', 'confidence'], 'diagnostics')
    if data[['dataset', 'true_label', 'predicted_label', 'confidence']].isna().any().any():
        raise ValueError('diagnostics contains missing labels or confidence')
    if not np.isfinite(data.confidence).all() or not data.confidence.between(0, 1).all():
        raise ValueError('confidence must be finite and in [0, 1]')
    data['true_label'] = data.true_label.astype(str)
    data['predicted_label'] = data.predicted_label.astype(str)
    data['error'] = data.true_label.ne(data.predicted_label)
    return data


def _embedding_axes(ax, coords):
    # L-shaped scale-free arrows are conventional for a qualitative UMAP view.
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.12)
    ax.set(xticks=[], yticks=[])
    ax.spines[:].set_visible(False)
    for end in [(0.19, 0.04), (0.04, 0.19)]:
        ax.annotate('', xy=end, xytext=(0.04, 0.04), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', linewidth=1.2, color='#60717E'))
    ax.text(0.12, -0.025, 'UMAP 1', transform=ax.transAxes, ha='center', fontsize=13)
    ax.text(-0.015, 0.12, 'UMAP 2', transform=ax.transAxes, va='center', rotation=90, fontsize=13)


def _point_size(n):
    return float(np.clip(650 / np.sqrt(max(n, 1)), 3, 68))


def plot_overview(composition, out_dir='figures', palette=None, save_pdf=False):
    """Show full-data abundance and composition; no workflow schematic."""
    _style()
    _require(composition, ['dataset', 'label', 'count'], 'composition')
    if composition['count'].isna().any() or (composition['count'] < 0).any():
        raise ValueError('Composition counts must be complete and nonnegative')
    datasets = [d for d in DATASET_SHORT if d in set(composition.dataset)]
    datasets += sorted(set(composition.dataset) - set(datasets))
    labels = _ordered_labels(composition.label)
    counts = composition.pivot_table(index='label', columns='dataset', values='count', aggfunc='sum', fill_value=0).reindex(index=labels, columns=datasets)
    totals = counts.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError('Each dataset must contain cells')
    fractions = counts.div(totals, axis=1)
    fig = plt.figure(figsize=(13.5, 10.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.30], wspace=0.58)
    ax = fig.add_subplot(grid[0])
    positions = np.arange(len(datasets))
    ax.barh(positions, totals.values / 1000, height=0.47,
            color=[DATASET_COLORS.get(d, '#657B91') for d in datasets])
    ax.set_yticks(positions, [DATASET_SHORT.get(d, d) for d in datasets], fontsize=16)
    ax.invert_yaxis()
    for index, name in enumerate(datasets):
        ax.text(totals[name] / 1000 + 3, index, f'{int(totals[name]):,}', va='center', fontsize=18, fontweight='semibold')
        ax.text(2, index + 0.32, f'{int((counts[name] > 0).sum())} subclasses', va='top', fontsize=14, color='#60717E')
    ax.set_ylim(len(datasets) - 0.4, -0.8)
    ax.set_xlim(0, totals.max() / 1000 * 1.42)
    ax.set_xlabel('Cells (thousands)', labelpad=12)
    ax.set_title('a  Dataset size', loc='left', fontweight='semibold', pad=22)
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax = fig.add_subplot(grid[1])
    xx, yy = np.meshgrid(np.arange(len(datasets)), np.arange(len(labels)))
    values = fractions.to_numpy()
    # Square-root area scaling reveals rare observed types without inventing cells.
    # Exactly zero abundance still has zero area; the legend states the scale.
    dot_colors = [DATASET_COLORS.get(d, '#657B91') for d in datasets] * len(labels)
    ax.scatter(xx.ravel(), yy.ravel(), s=600 * np.sqrt(values.ravel()), c=dot_colors,
               edgecolors='none', zorder=3)
    ax.set_xticks(range(len(datasets)), [DATASET_SHORT.get(d, d) for d in datasets], fontsize=16)
    ax.set_yticks(range(len(labels)), labels, fontsize=14)
    ax.set(xlim=(-0.65, len(datasets) - 0.35), ylim=(len(labels) - 0.4, -0.6))
    ax.tick_params(length=0, pad=8)
    ax.spines[:].set_visible(False)
    ax.grid(axis='y', color='#EDF0F3', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('b  Cell-type composition', loc='left', fontweight='semibold', pad=22)
    legend = [ax.scatter([], [], s=600 * np.sqrt(value), c='#60717E', label=f'{100 * value:g}%') for value in [0.001, 0.01, 0.10, 0.40]]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.50, -0.075), ncol=4,
              frameon=False, fontsize=14, handletextpad=0.4, columnspacing=0.85,
              title='Cell fraction · square-root area scale', title_fontsize=14)
    fig.suptitle('Three views of the mouse motor cortex', fontsize=23, fontweight='semibold', y=0.98)
    fig.text(0.5, 0.925, 'Full dataset composition', ha='center', fontsize=15, color='#60717E')
    fig.subplots_adjust(left=0.10, right=0.96, top=0.84, bottom=0.15)
    return [_save(fig, out_dir, '01_dataset_landscape', save_pdf, note='')]


def plot_exploratory_atlas(atlas, out_dir='figures', save_pdf=False):
    """Joint unsupervised coordinates only: labels are original annotations, never predictions."""
    _style()
    _require(atlas, ['dataset', 'class_label', 'umap1', 'umap2'], 'exploratory atlas')
    if not np.isfinite(atlas[['umap1', 'umap2']]).all().all():
        raise ValueError('Atlas coordinates must be finite')
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 9.4))
    classes = {'Glutamatergic': '#D4824C', 'GABAergic': '#3E9485', 'Non-Neuronal': '#7972AA'}
    for ax, key, colors, title in [(axes[0], 'dataset', DATASET_COLORS, 'a  Data source'),
                                    (axes[1], 'class_label', classes, 'b  Major cell classes')]:
        frame = atlas.sample(frac=1, random_state=42)
        unknown = set(frame[key]) - set(colors)
        if unknown:
            raise ValueError(f'Unknown atlas categories: {unknown}')
        ax.scatter(frame.umap1, frame.umap2, c=frame[key].map(colors), s=8, alpha=0.72, linewidths=0, rasterized=True)
        _embedding_axes(ax, frame)
        ax.set_title(title, loc='left', fontsize=19, pad=16, fontweight='semibold')
        handles = [Line2D([], [], marker='o', linestyle='', color=color, markersize=8,
                          label=DATASET_SHORT.get(label, label)) for label, color in colors.items() if label in set(frame[key])]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=1 if key == 'class_label' else 3,
                  frameon=False, fontsize=15, handletextpad=0.5, columnspacing=1.0)
    fig.suptitle('A shared view of transcriptomic diversity', fontsize=23, fontweight='semibold', y=0.99)
    fig.text(0.5, 0.92, f'{len(atlas):,} sampled cells · Joint UMAP · No batch correction', ha='center', fontsize=15, color='#60717E')
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.25, wspace=0.20)
    return [_save(fig, out_dir, '01_exploratory_atlas', save_pdf, note='Exploratory expression view; not a classification evaluation')]


def plot_annotation(diagnostics, out_dir='figures', palette=None, confidence_threshold=0.5, save_pdf=False):
    _style()
    data = _diagnostics(diagnostics)
    _require(data, ['umap1', 'umap2'], 'diagnostics')
    if not np.isfinite(data[['umap1', 'umap2']]).all().all():
        raise ValueError('Annotation UMAP coordinates must be finite')
    colors = _palette(pd.concat([data.true_label, data.predicted_label]), palette)
    paths = []
    for dataset, frame in data.groupby('dataset', sort=False):
        labels = _ordered_labels(pd.concat([frame.true_label, frame.predicted_label]))
        legend_rows = int(np.ceil(len(labels) / 5))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5 + legend_rows * 0.35))
        for ax, column, title in zip(axes, ['true_label', 'predicted_label'], ['a  Reference labels', 'b  Ensemble predictions']):
            ax.scatter(frame.umap1, frame.umap2, c=frame[column].map(colors), s=_point_size(len(frame)), linewidths=0.4, edgecolors='white', rasterized=True)
            _embedding_axes(ax, frame)
            ax.set_title(title, loc='left', fontweight='semibold', pad=18)
        handles = [Line2D([], [], linestyle='', marker='o', markersize=8, color=colors[label], label=label) for label in labels]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.50, 0.01), ncol=5, frameon=False, fontsize=14,
                   handletextpad=0.4, columnspacing=1.9, labelspacing=0.7)
        fig.suptitle(f'{DATASET_SHORT.get(dataset, dataset)} · Cell-type annotation', fontsize=23, fontweight='semibold', y=0.99)
        fig.text(0.5, 0.92, f'{len(frame):,} displayed test cells · Same UMAP coordinates in both panels', ha='center', fontsize=15, color='#60717E')
        fig.subplots_adjust(left=0.06, right=0.98, top=0.81, bottom=(1.0 + 0.36 * legend_rows) / fig.get_figheight(), wspace=0.20)
        paths.append(_save(fig, out_dir, f'02_annotation_{_slug(dataset)}', save_pdf))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.7))
        cmap = LinearSegmentedColormap.from_list('confidence', ['#E8E7EF', '#9181BB', '#433779'])
        im = axes[0].scatter(frame.umap1, frame.umap2, c=frame.confidence, cmap=cmap, vmin=0, vmax=1,
                             s=_point_size(len(frame)), linewidths=0.4, edgecolors='white', rasterized=True)
        bar = fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.025, ticks=[0, 0.5, 1])
        bar.set_label('Max. ensemble probability', fontsize=14, labelpad=12)
        axes[0].set_title('a  Confidence', loc='left', fontweight='semibold', pad=18)
        for error, color, name in [(False, '#9CADB7', 'Correct'), (True, '#C45568', 'Incorrect')]:
            part = frame.loc[frame.error.eq(error)]
            axes[1].scatter(part.umap1, part.umap2, s=_point_size(len(frame)), color=color, edgecolors='white', linewidths=0.4,
                            label=f'{name} ({len(part)})', rasterized=True)
        axes[1].set_title('b  Classification errors', loc='left', fontweight='semibold', pad=18)
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=15)
        for ax in axes:
            _embedding_axes(ax, frame)
        fig.suptitle(f'{DATASET_SHORT.get(dataset, dataset)} · Where predictions are uncertain', fontsize=23, fontweight='semibold', y=0.99)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.81, bottom=0.19, wspace=0.27)
        paths.append(_save(fig, out_dir, f'02_uncertainty_{_slug(dataset)}', save_pdf))
    return paths


def plot_confusion(diagnostics, out_dir='figures', palette=None, save_pdf=False):
    _style()
    data = _diagnostics(diagnostics)
    paths = []
    cmap = LinearSegmentedColormap.from_list('confusion', ['white', '#D2E6ED', '#5D9DB9', '#20536E'])
    cmap.set_bad('white')
    for dataset, frame in data.groupby('dataset', sort=False):
        labels = _ordered_labels(pd.concat([frame.true_label, frame.predicted_label]))
        counts = pd.crosstab(frame.true_label, frame.predicted_label).reindex(index=labels, columns=labels, fill_value=0)
        totals = counts.sum(axis=1)
        fractions = counts.div(totals.replace(0, np.nan), axis=0)
        fig, ax = plt.subplots(figsize=(12.5, 11.3))
        im = ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right', rotation_mode='anchor', fontsize=14)
        ax.set_yticks(range(len(labels)), [f'{label}  ({totals[label]})' for label in labels], fontsize=14)
        ax.tick_params(length=0, pad=8)
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which='minor', color='#EEF1F3', linewidth=0.5)
        ax.tick_params(which='minor', length=0)
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = fractions.iloc[row, col]
                if np.isfinite(value) and value >= 0.10:
                    ax.text(col, row, f'{value:.0%}', ha='center', va='center', fontsize=13,
                            color='white' if value >= 0.65 else '#304553')
        ax.set_xlabel('Predicted subclass', labelpad=18)
        ax.set_ylabel('Reference subclass (test cells)', labelpad=18)
        ax.set_title(f'{DATASET_SHORT.get(dataset, dataset)} · Classification structure', fontsize=23, fontweight='semibold', pad=28)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1])
        cb.set_label('Fraction within reference subclass', fontsize=15, labelpad=14)
        cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.tight_layout(pad=1.4)
        paths.append(_save(fig, out_dir, f'07_confusion_{_slug(dataset)}', save_pdf))
    return paths


def plot_markers(marker_table, out_dir='figures', palette=None, save_pdf=False):
    """Actual training-cell expression; dot area = fraction, color = mean log expression."""
    _style()
    _require(marker_table, ['dataset', 'label', 'gene', 'mean_expression', 'fraction_expressing'], 'markers')
    if marker_table[['mean_expression', 'fraction_expressing']].isna().any().any():
        raise ValueError('Marker measurements must be complete')
    if not marker_table.fraction_expressing.between(0, 1).all():
        raise ValueError('Marker fractions must be in [0, 1]')
    if marker_table.duplicated(['dataset', 'label', 'gene']).any():
        raise ValueError('Duplicate marker measurements')
    genes = [g for g in ['Slc17a7', 'Cux2', 'Rorb', 'Bcl11b', 'Foxp2', 'Gad1', 'Sst', 'Pvalb', 'Vip', 'Aqp4', 'Pdgfra', 'Mbp'] if g in set(marker_table.gene)]
    if not genes:
        raise ValueError('No requested marker genes available')
    cmap = LinearSegmentedColormap.from_list('markers', ['#E6EBF1', '#AEA0C6', '#875793', '#492B60'])
    vmax = max(float(marker_table.mean_expression.max()), 1e-9)
    paths = []
    for dataset, frame in marker_table.groupby('dataset', sort=False):
        frame = frame.loc[frame.gene.isin(genes)]
        labels = _ordered_labels(frame.label)
        fig, ax = plt.subplots(figsize=(13.6, max(7.5, len(labels) * 0.38 + 2)))
        x = frame.gene.map({gene: i for i, gene in enumerate(genes)})
        y = frame.label.map({label: i for i, label in enumerate(labels)})
        dots = ax.scatter(x, y, c=frame.mean_expression, s=230 * frame.fraction_expressing,
                          cmap=cmap, vmin=0, vmax=vmax, edgecolors='none', zorder=3)
        ax.set_xticks(range(len(genes)), genes, rotation=50, ha='right', rotation_mode='anchor', fontsize=15, fontstyle='italic')
        ax.set_yticks(range(len(labels)), labels, fontsize=15)
        ax.set(xlim=(-0.65, len(genes) - 0.35), ylim=(len(labels) - 0.45, -0.65))
        ax.tick_params(length=0, pad=9)
        ax.spines[:].set_visible(False)
        ax.grid(color='#EBEEF1', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(f'{DATASET_SHORT.get(dataset, dataset)} · Marker signatures', fontsize=23, fontweight='semibold', pad=28)
        fig.subplots_adjust(left=0.16, right=0.77, bottom=0.18, top=0.89)
        cax = fig.add_axes([0.82, 0.46, 0.018, 0.31])
        cb = fig.colorbar(dots, cax=cax)
        cb.set_label('Mean log expression', fontsize=15, labelpad=13)
        cb.ax.yaxis.set_major_locator(MaxNLocator(4))
        handles = [ax.scatter([], [], s=230 * value, color='#8E7CA5', label=f'{value:.0%}') for value in [0.25, 0.5, 1.0]]
        ax.legend(handles=handles, title='Cells expressing', title_fontsize=14, loc='lower left',
                  bbox_to_anchor=(1.045, 0.01), frameon=False, fontsize=14, labelspacing=1.0)
        paths.append(_save(fig, out_dir, f'08_markers_{_slug(dataset)}', save_pdf))
    return paths


def plot_marker_expression(diagnostics, out_dir='figures', markers=('Slc17a7', 'Gad1', 'Aqp4', 'Mbp'), save_pdf=False):
    """Observed expression on supplied coordinates; a shared color scale for each gene."""
    _style()
    data = (pd.concat([frame.assign(dataset=name) for name, frame in diagnostics.items()], ignore_index=True)
            if isinstance(diagnostics, dict) else diagnostics.copy())
    columns = ['marker_' + gene for gene in markers]
    _require(data, ['dataset', 'umap1', 'umap2', *columns], 'marker coordinates')
    if not np.isfinite(data[['umap1', 'umap2', *columns]].to_numpy()).all() or (data[columns] < 0).any().any():
        raise ValueError('Marker expression must be measured, finite and nonnegative')
    cmap = LinearSegmentedColormap.from_list('expression', ['#E3E7EB', '#A6A0C6', '#7461A4', '#3C266D'])
    maxima = data[columns].max()
    paths = []
    for dataset, frame in data.groupby('dataset', sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10.8))
        for index, (gene, column) in enumerate(zip(markers, columns)):
            ax = axes.flat[index]
            ordered = frame.sort_values(column)
            im = ax.scatter(ordered.umap1, ordered.umap2, c=ordered[column], s=min(40, _point_size(len(frame)) / 2),
                            cmap=cmap, vmin=0, vmax=max(float(maxima[column]), 1e-9), linewidths=0, rasterized=True)
            _embedding_axes(ax, frame)
            ax.set_title(gene, fontsize=20, fontstyle='italic', fontweight='semibold', loc='left', pad=12)
            bar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
            bar.ax.yaxis.set_major_locator(MaxNLocator(3))
            bar.ax.tick_params(labelsize=13)
        fig.suptitle(f'{DATASET_SHORT.get(dataset, dataset)} · Marker expression', fontsize=23, fontweight='semibold', y=0.985)
        fig.text(0.5, 0.935, 'Log-normalized expression · Shared per-gene scales across sources', ha='center', fontsize=15, color='#60717E')
        fig.subplots_adjust(left=0.06, right=0.95, top=0.86, bottom=0.06, hspace=0.25, wspace=0.23)
        paths.append(_save(fig, out_dir, f'10_marker_expression_{_slug(dataset)}', save_pdf))
    return paths


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def _metric_dataset_order(frame):
    available = list(frame["dataset"].drop_duplicates())
    preferred = ["10X_cells_v2_AIBS", "10X_cells_v3_AIBS", "SMARTer_cells_MOp"]
    return [name for name in preferred if name in available] + [name for name in available if name not in preferred]


def _metric_repeat_note(frame):
    groups = [name for name in ["dataset", "feature", "regime", "model", "split"] if name in frame]
    counts = frame.groupby(groups)["seed"].nunique() if groups else pd.Series([frame["seed"].nunique()])
    if counts.max() == 1:
        return "One seed"
    if counts.nunique() == 1:
        return f"Mean ± SD · {int(counts.iloc[0])} seeds"
    return "Mean ± SD where repeated seeds are available"


def _metric_context_note(frame, regime, feature=None):
    context = {"full": "All subclasses", "matched": "Matched classes & training counts"}.get(regime, regime)
    parts = [context]
    if feature is not None:
        parts.append({"A_raw": "Raw features", "B_log": "Log features",
                      "C_hvg2000": "2,000 HVGs", "D_hvg4000": "4,000 HVGs"}.get(feature, feature))
    return " · ".join(parts + [_metric_repeat_note(frame)])


def _metric_summary(frame, by, metric):
    return frame.groupby(by, sort=False)[metric].agg(["mean", "std", "count"])


def _metric_axes(ax, ylabel):
    ax.set_facecolor("white")
    ax.set_ylabel(ylabel, fontsize=15, labelpad=9)
    ax.tick_params(axis="both", labelsize=13, length=4, color="#56616B")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#78848E")
    ax.grid(axis="y", color="#E8EDF1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _metric_score_limits(axes, frame, metrics):
    for ax, metric in zip(axes, metrics):
        values = frame[metric].dropna().to_numpy()
        if not len(values):
            raise ValueError(f"No finite {metric} observations")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"{metric} must contain scores in [0, 1]")
        groups = [name for name in ["dataset", "feature", "regime", "model", "split"] if name in frame]
        summary = _metric_summary(frame, groups, metric)
        errors = summary["std"].fillna(0)
        lower = min(float(values.min()), float((summary["mean"] - errors).min()))
        upper = max(float(values.max()), float((summary["mean"] + errors).max()))
        # Keep every SD interval visible, even if it extends past the score bounds.
        margin = max(0.07, 0.16 * float(np.ptp(values)))
        lower_limit = max(0.0, lower - margin) if lower >= 0 else lower - margin
        upper_limit = min(1.03, upper + margin) if upper <= 1 else upper + margin
        ax.set_ylim(lower_limit, upper_limit)


def _metric_title(fig, title, note):
    fig.set_facecolor("white")
    fig.suptitle(title, fontsize=20, fontweight="semibold", y=0.99)
    fig.text(0.5, 0.925, note, ha="center", va="center", fontsize=13, color="#56616B")


def _metric_models_figure(data, title, note=None):
    datasets = _metric_dataset_order(data)
    models = [model for model in MODEL_ORDER if model in set(data["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    offsets = np.linspace(-0.22, 0.22, len(models)) if len(models) > 1 else [0]
    for ax, metric, panel in zip(axes, ["accuracy", "macro_f1"], ["a", "b"]):
        _metric_axes(ax, METRIC_NAMES[metric])
        for offset, model in zip(offsets, models):
            frame = data.loc[data["model"].eq(model)]
            summary = _metric_summary(frame, "dataset", metric)
            for position, dataset in enumerate(datasets):
                if dataset not in summary.index:
                    continue
                item = summary.loc[dataset]
                error = float(item["std"]) if item["count"] > 1 else None
                ax.errorbar(position + offset, item["mean"], yerr=error, fmt="o", markersize=9,
                            color=MODEL_COLORS[model], markeredgecolor="white", markeredgewidth=0.8,
                            elinewidth=1.8, capsize=4, capthick=1.5, zorder=3)
        ax.set_xticks(range(len(datasets)), [DATASET_SHORT.get(dataset, dataset) for dataset in datasets])
        ax.set_xlim(-0.55, len(datasets) - 0.45)
        ax.set_title(f"{panel}  {METRIC_NAMES[metric]}", loc="left", fontsize=17, pad=14)
    _metric_score_limits(axes, data, ["accuracy", "macro_f1"])
    handles = [Line2D([], [], linestyle="", marker="o", markersize=9, color=MODEL_COLORS[model], label=MODEL_NAMES[model]) for model in models]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=len(handles), frameon=False, fontsize=14, handletextpad=0.5, columnspacing=2.4)
    _metric_title(fig, title, note if note is not None else _metric_repeat_note(data))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.78, wspace=0.30)
    return fig


def plot_performance(metrics, per_class, out_dir="figures", feature="B_log", regime="full", split="test", palette=None, save_pdf=False):
    """Two large overview panels, then one box/scatter figure per dataset."""
    _style()
    _require(metrics, ["dataset", "seed", "model", "accuracy", "macro_f1"], "metrics")
    data = _select(metrics, feature=feature, regime=regime, split=split)
    _require(per_class, ["dataset", "seed", "model", "label", "f1"], "per_class")
    filters = dict(feature=feature, regime=regime)
    if "split" in per_class:
        filters["split"] = split
    types = _select(per_class, name="per_class", **filters)
    types["label"] = types["label"].astype(str)
    colors = _palette(types["label"], palette)
    tag = f"{_slug(feature)}_{_slug(regime)}_{_slug(split)}"
    fig = _metric_models_figure(data, "Cell-type annotation", _metric_context_note(data, regime, feature))
    paths = [_save(fig, out_dir, f"03_performance_{tag}", save_pdf)]

    for dataset in _metric_dataset_order(data):
        frame = types.loc[types["dataset"].eq(dataset)]
        if frame.empty:
            raise ValueError(f"No per-class results for {dataset}")
        means = frame.groupby(["model", "label"], as_index=False)["f1"].mean()
        models = [model for model in MODEL_ORDER if model in set(means["model"])]
        labels = [label for label in LABEL_ORDER if label in set(frame["label"])]
        labels += sorted(set(frame["label"]) - set(labels))
        legend_rows = int(np.ceil(len(labels) / 4))
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.9 + 0.30 * legend_rows))
        ax = axes[0]
        values = [means.loc[means["model"].eq(model), "f1"].to_numpy() for model in models]
        boxes = ax.boxplot(values, positions=np.arange(len(models)), widths=0.50, patch_artist=True,
                           showfliers=False, medianprops={"color": "#243745", "linewidth": 2},
                           whiskerprops={"color": "#80909C", "linewidth": 1.3},
                           capprops={"color": "#80909C", "linewidth": 1.3})
        for box in boxes["boxes"]:
            box.set(facecolor="white", edgecolor="#80909C", linewidth=1.3)
        for position, model in enumerate(models):
            points = means.loc[means["model"].eq(model)].sort_values("label")
            jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(points))
            ax.scatter(position + jitter, points["f1"], c=[colors[label] for label in points["label"]],
                       s=45, alpha=0.90, edgecolor="white", linewidth=0.5, zorder=3)
        _metric_axes(ax, "Cell-type F1")
        ax.set_xticks(range(len(models)), [MODEL_NAMES[model] for model in models])
        ax.set_ylim(-0.04, 1.05)
        ax.set_title("a  Across cell types", loc="left", fontsize=17, pad=16)

        keys = ["dataset", "seed", "regime", "feature", "label"] + (["split"] if "split" in frame else [])
        baseline = frame.loc[frame["model"].eq("single"), keys + ["f1"]]
        ensemble = frame.loc[frame["model"].eq("ensemble"), keys + ["f1"]]
        paired = baseline.merge(ensemble, on=keys, suffixes=("_single", "_ensemble"), validate="one_to_one")
        if paired.empty:
            raise ValueError(f"{dataset}: no paired single/ensemble F1 results")
        ax = axes[1]
        ax.plot([0, 1], [0, 1], color="#95A1AA", linestyle="--", linewidth=1.3, zorder=1)
        ax.scatter(paired["f1_single"], paired["f1_ensemble"], c=[colors[label] for label in paired["label"]],
                   s=54, alpha=0.82, edgecolor="white", linewidth=0.6, zorder=3)
        _metric_axes(ax, "Ensemble F1")
        ax.set_xlabel("Single kNN F1", fontsize=15, labelpad=9)
        ax.set(xlim=(-0.04, 1.05), ylim=(-0.04, 1.05))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_title("b  Paired comparison", loc="left", fontsize=17, pad=16)
        handles = [Line2D([], [], linestyle="", marker="o", markersize=6.5, color=colors[label], label=label) for label in labels]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4,
                   frameon=False, fontsize=13, columnspacing=1.8, handletextpad=0.4)
        _metric_title(fig, f"{DATASET_SHORT.get(dataset, dataset)} · Cell-type performance", _metric_context_note(frame, regime, feature))
        # Additional canvas and bottom reserve keep the axes tall while separating
        # the scatter x-label from the first row of the cell-type legend.
        bottom = (1.20 + legend_rows * 0.28) / fig.get_figheight()
        fig.subplots_adjust(left=0.08, right=0.97, bottom=bottom, top=0.79, wspace=0.32)
        paths.append(_save(fig, out_dir, f"04_celltype_f1_{_slug(dataset)}_{tag}", save_pdf))
    return paths


def plot_feature_comparison(metrics, out_dir="figures", regime="full", split="test", model="ensemble", save_pdf=False):
    """A-D feature settings with measured seed means and optional SD intervals."""
    _style()
    data = _select(metrics, regime=regime, split=split, model=model)
    _require(data, ["dataset", "seed", "feature", "accuracy", "macro_f1"], "metrics")
    datasets = _metric_dataset_order(data)
    dataset_colors = ["#397CA8", "#D08047", "#7764A6", "#419B83"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    for ax, metric, panel in zip(axes, ["accuracy", "macro_f1"], ["a", "b"]):
        _metric_axes(ax, METRIC_NAMES[metric])
        for index, dataset in enumerate(datasets):
            frame = data.loc[data["dataset"].eq(dataset)]
            summary = _metric_summary(frame, "feature", metric).reindex(FEATURE_ORDER)
            keep = summary["mean"].notna()
            x = np.arange(len(FEATURE_ORDER))[keep]
            selected = summary.loc[keep]
            color = dataset_colors[index % len(dataset_colors)]
            ax.plot(x, selected["mean"], color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            repeated = selected["count"] > 1
            if repeated.any():
                ax.errorbar(x[repeated], selected.loc[repeated, "mean"], yerr=selected.loc[repeated, "std"],
                            fmt="none", ecolor=color, elinewidth=1.6, capsize=4, zorder=2)
        ax.set_xticks(range(len(FEATURE_ORDER)), [FEATURE_NAMES[feature] for feature in FEATURE_ORDER])
        ax.set_xlim(-0.20, len(FEATURE_ORDER) - 0.80)
        ax.set_title(f"{panel}  {METRIC_NAMES[metric]}", loc="left", fontsize=17, pad=14)
    _metric_score_limits(axes, data, ["accuracy", "macro_f1"])
    handles = [Line2D([], [], color=dataset_colors[index % len(dataset_colors)], marker="o", markersize=7,
                      linewidth=2, label=DATASET_SHORT.get(dataset, dataset)) for index, dataset in enumerate(datasets)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=len(datasets), frameon=False, fontsize=14, columnspacing=2.4)
    _metric_title(fig, "Feature comparison", f"{MODEL_NAMES.get(model, model)} · {_metric_context_note(data, regime)}")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.78, wspace=0.30)
    return [_save(fig, out_dir, f"05_feature_comparison_{_slug(model)}_{_slug(regime)}_{_slug(split)}", save_pdf)]


def plot_ablation(metrics, out_dir="figures", feature="B_log", regime="full", split="test", save_pdf=False):
    """Trace each data source through single-k, multi-k and full-ensemble models."""
    _style()
    data = _select(metrics, feature=feature, regime=regime, split=split)
    _require(data, ["dataset", "seed", "model", "accuracy", "macro_f1"], "metrics")
    datasets = _metric_dataset_order(data)
    for dataset in datasets:
        missing = set(MODEL_ORDER) - set(data.loc[data["dataset"].eq(dataset), "model"])
        if missing:
            raise ValueError(f"{dataset}: ablation lacks {sorted(missing)}")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    positions = np.arange(len(MODEL_ORDER))
    for ax, metric, panel in zip(axes, ["accuracy", "macro_f1"], ["a", "b"]):
        _metric_axes(ax, METRIC_NAMES[metric])
        for dataset in datasets:
            frame = data.loc[data["dataset"].eq(dataset)]
            summary = _metric_summary(frame, "model", metric).reindex(MODEL_ORDER)
            color = DATASET_COLORS[dataset]
            ax.plot(positions, summary["mean"], color=color, linewidth=2, marker="o",
                    markersize=8, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            repeated = summary["count"] > 1
            if repeated.any():
                ax.errorbar(positions[repeated], summary.loc[repeated, "mean"],
                            yerr=summary.loc[repeated, "std"], fmt="none", ecolor=color,
                            elinewidth=1.6, capsize=4, zorder=2)
        ax.set_xticks(positions, [MODEL_NAMES[model] for model in MODEL_ORDER])
        ax.set_xlim(-0.18, len(MODEL_ORDER) - 0.82)
        ax.set_title(f"{panel}  {METRIC_NAMES[metric]}", loc="left", fontsize=17, pad=14)
    _metric_score_limits(axes, data, ["accuracy", "macro_f1"])
    handles = [Line2D([], [], color=DATASET_COLORS[dataset], marker="o", markersize=7,
                      linewidth=2, label=DATASET_SHORT.get(dataset, dataset)) for dataset in datasets]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=len(datasets),
               frameon=False, fontsize=14, columnspacing=2.4)
    _metric_title(fig, "Ensemble ablation", _metric_context_note(data, regime, feature))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.78, wspace=0.30)
    return [_save(fig, out_dir, f"06_ensemble_ablation_{_slug(feature)}_{_slug(regime)}_{_slug(split)}", save_pdf)]


def plot_confidence(diagnostics, out_dir="figures", confidence_threshold=0.5, save_pdf=False):
    """One readable confidence figure per dataset; coverage uses all supplied test cells."""
    _style()
    data = _diagnostics(diagnostics)
    paths = []
    correct_color, error_color = "#568FAB", "#BC6879"
    coverage_color, accuracy_color = "#A9C5D5", "#347FA2"
    for dataset in _metric_dataset_order(data):
        frame = data.loc[data["dataset"].eq(dataset)]
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
        ax = axes[0]
        for error, color in [(False, correct_color), (True, error_color)]:
            values = frame.loc[frame["error"].eq(error), "confidence"]
            ax.hist(values, bins=np.linspace(0, 1, 21), color=color, alpha=0.65, edgecolor="white", linewidth=0.6)
        ax.axvline(confidence_threshold, color="#58646D", linestyle="--", linewidth=1.5)
        _metric_axes(ax, "Test cells")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_xlabel("Confidence", fontsize=15, labelpad=9)
        ax.set_xlim(0, 1)
        ax.set_title("a  Prediction confidence", loc="left", fontsize=17, pad=44)
        ax.legend(handles=[Patch(facecolor=correct_color, alpha=0.65, label="Correct"), Patch(facecolor=error_color, alpha=0.65, label="Incorrect")],
                  frameon=False, fontsize=13, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)

        ax = axes[1]
        for position, mask in enumerate([frame["confidence"].gt(confidence_threshold), frame["confidence"].le(confidence_threshold)]):
            n = int(mask.sum())
            ax.bar(position - 0.17, n / len(frame), width=0.28, color=coverage_color, zorder=3)
            if n:
                ax.bar(position + 0.17, 1 - frame.loc[mask, "error"].mean(), width=0.28, color=accuracy_color, zorder=3)
            ax.text(position, 1.035, f"n = {n:,}", ha="center", va="bottom", fontsize=13, color="#56616B")
        _metric_axes(ax, "Fraction")
        ax.set_xticks([0, 1], [f"High (> {confidence_threshold:g})", f"Low (≤ {confidence_threshold:g})"])
        ax.set_ylim(0, 1.14)
        ax.set_title("b  Coverage and accuracy", loc="left", fontsize=17, pad=44)
        ax.legend(handles=[Patch(facecolor=coverage_color, label="Coverage"), Patch(facecolor=accuracy_color, label="Accuracy")],
                  frameon=False, fontsize=13, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
        fig.set_facecolor("white")
        fig.suptitle(f"{DATASET_SHORT.get(dataset, dataset)} · Confidence", fontsize=20, fontweight="semibold", y=0.99)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.75, wspace=0.30)
        paths.append(_save(fig, out_dir, f"09_confidence_{_slug(dataset)}", save_pdf))
    return paths


import os
from pathlib import Path


def _configure_atlas_environment(out_dir):
    cache_root = Path(out_dir).resolve().parent
    for variable, folder in (("NUMBA_CACHE_DIR", "q21_numba_cache"),
                             ("MPLCONFIGDIR", "q21_mpl_cache")):
        path = cache_root / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


def build_exploratory_atlas(data_dir, out_dir, max_cells_per_dataset=1500,
                            seed=42, n_hvg=2000, n_svd=30):
    """Write measured-cell coordinates/markers for descriptive visualization.

    Requires the existing inspect_dataset, common_gene_names, stratified_subset,
    load_expression, log_normalize and hvg_ranking functions in module globals.
    Labels only stratify sampling and annotate the result. Pooled HVGs, SVD and
    UMAP belong exclusively to this atlas; no classifier or metric is touched.
    """
    import gc
    import json
    import time
    from importlib.metadata import version

    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from threadpoolctl import threadpool_limits

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_atlas_environment(out_dir)
    started = time.perf_counter()
    names = ("10X_cells_v3_AIBS", "10X_cells_v2_AIBS", "SMARTer_cells_MOp")
    markers = ("Slc17a7", "Gad1", "Aqp4", "Mbp")
    if max_cells_per_dataset < 1 or n_hvg < 2 or n_svd < 2:
        raise ValueError("Atlas sampling and embedding dimensions must be positive.")
    metas = [inspect_dataset(data_dir, name) for name in names]
    genes = common_gene_names(metas)
    marker_indices = pd.Index(genes).get_indexer(markers)
    if np.any(marker_indices < 0):
        missing = [marker for marker, index in zip(markers, marker_indices) if index < 0]
        raise ValueError(f"Atlas markers missing from common genes: {missing}")
    blocks, annotation_blocks, sampling = [], [], []
    for meta in metas:
        print(f"[atlas] Streaming {meta['name']}", flush=True)
        cells = meta["cells"]
        for column in ("cell_id", "class_label", "subclass_label"):
            if column not in cells or cells[column].isna().any() or cells[column].eq("").any():
                raise ValueError(f"{meta['name']}: missing atlas annotations in {column}.")
        rows = stratified_subset(cells["subclass_label"].to_numpy(),
                                 max_cells_per_dataset, seed, minimum=3)
        counts = load_expression(meta, rows, genes)
        log_counts = log_normalize(counts)
        del counts
        annotation = cells.iloc[rows][["cell_id", "class_label", "subclass_label"]].copy()
        annotation.insert(0, "dataset", meta["name"])
        # Only four columns are densified; the expression matrices stay sparse.
        observed_markers = log_counts[:, marker_indices].toarray()
        for index, marker in enumerate(markers):
            annotation[f"marker_{marker}"] = observed_markers[:, index]
        annotation_blocks.append(annotation.reset_index(drop=True))
        blocks.append(log_counts)
        sampling.append({
            "dataset": meta["name"], "available_cells": int(len(cells)),
            "sampled_cells": int(len(rows)),
            "subclass_counts": {str(key): int(value) for key, value in
                                annotation["subclass_label"].value_counts().sort_index().items()},
        })
        print(f"[atlas] {meta['name']}: {len(rows)} cells; "
              f"{log_counts.nnz:,} nonzeros; elapsed {time.perf_counter()-started:.1f}s", flush=True)
        del log_counts, annotation, observed_markers
        gc.collect()
    print("[atlas] Pooled HVG selection", flush=True)
    pooled = sparse.vstack(blocks, format="csr", dtype=np.float32)
    blocks.clear()
    del blocks
    gc.collect()
    ranking, scores = hvg_ranking(pooled)
    eligible = ranking[np.isfinite(scores[ranking])]
    if len(eligible) < n_hvg:
        raise ValueError(f"Only {len(eligible)} valid HVGs, fewer than requested {n_hvg}.")
    selected_genes = eligible[:n_hvg]
    selected = pooled[:, selected_genes].tocsr()
    del pooled, scores, ranking, eligible
    gc.collect()
    if min(selected.shape) <= n_svd:
        raise ValueError("Atlas sample/HVG dimensions are too small for requested SVD.")
    print(f"[atlas] TruncatedSVD({n_svd}) on {selected.shape}", flush=True)
    with threadpool_limits(limits=2):
        svd = TruncatedSVD(n_components=n_svd, n_iter=7, random_state=seed)
        latent = svd.fit_transform(selected).astype(np.float32, copy=False)
    del selected
    gc.collect()
    print("[atlas] Importing UMAP with workspace-local numba cache", flush=True)
    import umap
    print("[atlas] Fitting unsupervised UMAP", flush=True)
    umap_parameters = {
        "n_neighbors": 30, "min_dist": 0.3, "n_components": 2,
        "metric": "cosine", "random_state": int(seed),
        "transform_seed": int(seed), "n_jobs": 1,
    }
    with threadpool_limits(limits=2):
        coordinates = umap.UMAP(**umap_parameters).fit_transform(latent)
    if coordinates.shape != (sum(x["sampled_cells"] for x in sampling), 2):
        raise ValueError("Unexpected UMAP coordinate shape.")
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP produced nonfinite coordinates.")
    result = pd.concat(annotation_blocks, ignore_index=True)
    result["umap1"], result["umap2"] = coordinates[:, 0], coordinates[:, 1]
    result = result[["dataset", "cell_id", "class_label", "subclass_label", "umap1", "umap2"]
                    + [f"marker_{marker}" for marker in markers]]
    if result.duplicated(["dataset", "cell_id"]).any():
        raise ValueError("Duplicate sampled cell IDs within a dataset.")
    coordinate_path = out_dir / "atlas_coordinates.csv"
    result.to_csv(coordinate_path, index=False, encoding="utf-8")
    metadata = {
        "purpose": "Exploratory pooled UMAP of measured single-cell expression only",
        "classification_predictions_generated": False,
        "existing_model_metrics_modified": False,
        "labels_in_embedding_fit": False,
        "seed": int(seed), "maximum_cells_per_dataset": int(max_cells_per_dataset),
        "sampling": "Proportional subclass-stratified sampling, at least three per subclass",
        "datasets": sampling, "common_gene_count": int(len(genes)),
        "normalization": "Within common genes: library total to 10000 followed by log1p",
        "hvg_selection": "Pooled sampled cells; Seurat-style mean-bin standardized dispersion",
        "hvg_count": int(n_hvg), "hvg_gene_names": genes[selected_genes].tolist(),
        "svd": {"n_components": int(n_svd), "n_iter": 7,
                "random_state": int(seed), "centered": False,
                "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum())},
        "umap": umap_parameters,
        "batch_correction": "None; source differences can remain visible",
        "marker_columns": {f"marker_{marker}":
                           f"Observed {marker} expression after common-gene log normalization; "
                           "not imputed or predicted" for marker in markers},
        "coordinate_interpretation": "Joint expression embedding, not tissue coordinates",
        "software_versions": {package: version(package) for package in
                              ("numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "numba")},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "coordinate_file": coordinate_path.name,
    }
    (out_dir / "atlas_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[atlas] Wrote {len(result)} cells to {coordinate_path}; "
          f"elapsed {metadata['elapsed_seconds']:.1f}s", flush=True)
    return result, metadata




import argparse
import gc
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits


DATASET_NAMES = ('10X_cells_v3_AIBS', '10X_cells_v2_AIBS', 'SMARTer_cells_MOp')
METRICS = ('braycurtis', 'correlation', 'cosine', 'chebyshev', 'euclidean')
K_VALUES = (10, 20, 30, 40)
FEATURES = ('A_raw', 'B_log', 'C_hvg2000', 'D_hvg4000')
MARKERS = ('Slc17a7', 'Gad1', 'Gad2', 'Aqp4', 'Mbp', 'Pdgfra',
           'Sst', 'Pvalb', 'Vip', 'Reln', 'Cux2', 'Rorb', 'Bcl11b', 'Foxp2', 'Ctgf')


def stratified_subset(labels, limit, seed, minimum=3):
    """按类别比例抽样且为三分割保留最少细胞；limit=None 表示全部。"""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if np.any(counts < minimum):
        rare = dict(zip(classes[counts < minimum], counts[counts < minimum].tolist()))
        raise ValueError(f'以下类别不足 {minimum} 个细胞，不能完成独立三分割：{rare}')
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    if limit < minimum * len(classes):
        raise ValueError(f'抽样上限至少为 {minimum * len(classes)}，否则不能保留全部类别。')
    capacity = counts - minimum
    allocation = (limit - minimum * len(classes)) * capacity / capacity.sum()
    quotas = minimum + np.floor(allocation).astype(int)
    order = np.argsort(-(allocation - np.floor(allocation)), kind='stable')
    quotas[order[:limit - quotas.sum()]] += 1
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(labels == c), int(n), replace=False)
              for c, n in zip(classes, quotas)]
    return np.sort(np.concatenate(chosen))


def split_by_type(labels, seed):
    """每类约70/15/15；罕见类也至少各有一个train/val/test细胞。"""
    rng = np.random.default_rng(seed)
    parts = {'train': [], 'val': [], 'test': []}
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 3:
            raise ValueError(f'{label} 不足三个细胞。')
        n_val = max(1, int(round(0.15 * len(indices))))
        n_test = max(1, int(round(0.15 * len(indices))))
        n_train = len(indices) - n_val - n_test
        parts['train'].extend(indices[:n_train])
        parts['val'].extend(indices[n_train:n_train + n_val])
        parts['test'].extend(indices[n_train + n_val:])
    return {name: np.sort(values).astype(int) for name, values in parts.items()}


def matched_indices(labels, split, target_counts, seed):
    """各来源采用相同类别、相同的每类训练数；验证/测试与训练仍不重叠。"""
    rng = np.random.default_rng(seed)
    train = []
    for label, count in sorted(target_counts.items()):
        candidates = split['train'][labels[split['train']] == label]
        if count > len(candidates):
            raise ValueError(f'匹配训练数量超出 {label} 的可用样本。')
        train.extend(rng.choice(candidates, count, replace=False))
    common = list(target_counts)
    return {
        'train': np.sort(train).astype(int),
        'val': split['val'][np.isin(labels[split['val']], common)],
        'test': split['test'][np.isin(labels[split['test']], common)],
    }


def log_normalize(counts):
    """在共同基因池内按细胞总量缩放至10000，再log1p；保持稀疏。"""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError('共同基因中存在无有效表达的细胞，请先检查数据质量。')
    result = sparse.diags(10000.0 / totals).dot(counts).tocsr().astype(np.float32)
    np.log1p(result.data, out=result.data)
    return result


def hvg_ranking(log_train):
    """Seurat式均值分箱dispersion：仅在训练集估计，高变基因保留非负log表达。"""
    total = np.zeros(log_train.shape[1], dtype=np.float64)
    total_square = np.zeros_like(total)
    for start in range(0, log_train.shape[0], 512):
        block = log_train[start:start + 512].astype(np.float64)
        np.expm1(block.data, out=block.data)
        total += np.asarray(block.sum(axis=0)).ravel()
        np.square(block.data, out=block.data)
        total_square += np.asarray(block.sum(axis=0)).ravel()
    mean = total / log_train.shape[0]
    variance = total_square / log_train.shape[0] - mean ** 2
    variance = np.maximum(variance, 0)
    if log_train.shape[0] > 1:
        variance *= log_train.shape[0] / (log_train.shape[0] - 1)
    valid = (mean > 0) & (variance > 0)
    frame = pd.DataFrame({'mean': np.log1p(mean), 'dispersion': np.nan})
    frame.loc[valid, 'dispersion'] = np.log(variance[valid] / mean[valid])
    frame['bin'] = pd.cut(frame['mean'], bins=20, duplicates='drop')
    grouped = frame.groupby('bin', observed=True)['dispersion']
    average = grouped.transform('mean')
    deviation = grouped.transform('std')
    scores = ((frame['dispersion'] - average) / deviation).to_numpy()
    singleton = valid & deviation.isna().to_numpy()
    scores[singleton] = 1.0
    scores[~np.isfinite(scores)] = -np.inf
    order = np.argsort(-scores, kind='stable')
    return order, scores


def training_distance_scales(x_train, block_size):
    """精确遍历训练-训练距离上三角，固定五种尺度；不访问query或标签。"""
    maxima = {metric: 0.0 for metric in METRICS}
    n = x_train.shape[0]
    for start in range(0, n, block_size):
        left = x_train[start:start + block_size].toarray()
        for other in range(start, n, block_size):
            right = left if other == start else x_train[other:other + block_size].toarray()
            for metric in METRICS:
                distances = cdist(left, right, metric=metric)
                finite = distances[np.isfinite(distances)]
                if finite.size:
                    maxima[metric] = max(maxima[metric], float(finite.max()))
    # 相同训练向量的距离可能全为0；常量向量的相关距离可能全为NaN。
    return {metric: value if value > 1e-12 else 1.0 for metric, value in maxima.items()}


def knn_probabilities(x_train, y_train, x_query, classes, scales, block_size):
    """精确分块近邻：只保留最近40个，距离相同时按训练行序稳定选择。"""
    if x_train.shape[0] < max(K_VALUES):
        raise ValueError(f'训练集不足 {max(K_VALUES)} 个细胞，无法使用论文的全部k。')
    encoded = pd.Index(classes).get_indexer(y_train)
    if np.any(encoded < 0):
        raise ValueError('训练标签不在classes中。')
    probabilities = {(metric, k): np.zeros((x_query.shape[0], len(classes)), np.float32)
                     for metric in METRICS for k in K_VALUES}
    n_neighbors = max(K_VALUES)
    for start in range(0, x_query.shape[0], block_size):
        query = x_query[start:start + block_size].toarray()
        size = len(query)
        nearest_d = {m: np.full((size, n_neighbors), np.inf) for m in METRICS}
        nearest_i = {m: np.full((size, n_neighbors), x_train.shape[0], dtype=int) for m in METRICS}
        for other in range(0, x_train.shape[0], block_size):
            reference = x_train[other:other + block_size].toarray()
            reference_ids = np.arange(other, other + len(reference))
            for metric in METRICS:
                distance = cdist(query, reference, metric=metric)
                # 按论文把未定义距离置为最大距离；这里使用固定的训练尺度。
                distance[~np.isfinite(distance)] = scales[metric]
                np.maximum(distance, 0, out=distance)
                ids = np.broadcast_to(reference_ids, distance.shape)
                combined_d = np.concatenate([nearest_d[metric], distance], axis=1)
                combined_i = np.concatenate([nearest_i[metric], ids], axis=1)
                order = np.lexsort((combined_i, combined_d), axis=1)[:, :n_neighbors]
                nearest_d[metric] = np.take_along_axis(combined_d, order, axis=1)
                nearest_i[metric] = np.take_along_axis(combined_i, order, axis=1)
        for metric in METRICS:
            neighbor_labels = encoded[nearest_i[metric]]
            for k in K_VALUES:
                distance = nearest_d[metric][:, :k] / scales[metric]
                # 平移指数与原公式归一化后完全等价，可避免远query下权重全部下溢。
                weights = np.exp(-(distance - distance[:, :1]) / 0.1)
                weights /= weights.sum(axis=1, keepdims=True)
                votes = np.zeros((size, len(classes)), dtype=np.float64)
                np.add.at(votes, (np.repeat(np.arange(size), k), neighbor_labels[:, :k].ravel()), weights.ravel())
                probabilities[(metric, k)][start:start + size] = votes
    return probabilities


def average_probabilities(probabilities, keys):
    return np.mean([probabilities[key] for key in keys], axis=0)


def choose_models(probabilities, y_val, classes):
    """只用验证集选single与multi_k；完整20配置ensemble预先固定。"""
    def score(p):
        return f1_score(y_val, classes[p.argmax(axis=1)], labels=classes, average='macro', zero_division=0)
    single = max(probabilities, key=lambda key: score(probabilities[key]))
    multi_metric = max(METRICS, key=lambda metric: score(average_probabilities(
        probabilities, [(metric, k) for k in K_VALUES])))
    return {
        'single': [single],
        'multi_k': [(multi_metric, k) for k in K_VALUES],
        'ensemble': [(metric, k) for metric in METRICS for k in K_VALUES],
    }


def evaluate_probabilities(probability, truth, classes, context):
    prediction = classes[probability.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0)
    summary = dict(context, accuracy=float(accuracy_score(truth, prediction)),
                   macro_f1=float(f1.mean()), balanced_accuracy=float(recall.mean()))
    per_type = [dict(context, label=label, precision=float(p), recall=float(r),
                     f1=float(f), support=int(n))
                for label, p, r, f, n in zip(classes, precision, recall, f1, support)]
    return summary, per_type


def marker_summary(log_train, labels, genes, dataset):
    records = []
    for gene in MARKERS:
        positions = np.flatnonzero(genes == gene)
        if not len(positions):
            continue
        values = log_train[:, int(positions[0])].toarray().ravel()
        for label in np.unique(labels):
            group = values[labels == label]
            records.append(dict(dataset=dataset, label=label, gene=gene,
                                mean_expression=float(group.mean()),
                                fraction_expressing=float(np.mean(group > 0))))
    if not records:
        raise ValueError('共同基因池中没有指定marker，不能生成所设计的marker图。')
    return records


def umap_for_test(log_train, log_test, hvg_order, seed):
    """降维仅服务展示：训练HVG和SVD固定，再对测试细胞作无监督UMAP。"""
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError('绘制论文风格图需要 umap-learn；请先安装，不会用其他坐标冒充UMAP。') from error
    features = hvg_order[:min(2000, len(hvg_order))]
    dimensions = min(30, log_train.shape[0] - 1, len(features) - 1)
    reducer = TruncatedSVD(n_components=dimensions, random_state=seed)
    reducer.fit(log_train[:, features])
    test_embedding = reducer.transform(log_test[:, features])
    if len(test_embedding) < 4:
        raise ValueError('测试细胞过少，无法稳定绘制UMAP。')
    embedding = UMAP(n_neighbors=min(15, len(test_embedding) - 1), min_dist=0.25,
                          metric='euclidean', random_state=seed, n_jobs=1)
    return embedding.fit_transform(test_embedding)


def save_tables(output_dir, metrics, per_type, configuration_rows, model_settings):
    pd.DataFrame(metrics).to_csv(output_dir / 'metrics.csv', index=False)
    pd.DataFrame(per_type).to_csv(output_dir / 'per_class_metrics.csv', index=False)
    pd.DataFrame(configuration_rows).to_csv(output_dir / 'validation_configurations.csv', index=False)
    (output_dir / 'model_settings.json').write_text(
        json.dumps(model_settings, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_experiment(metadata, seeds, max_cells):
    plans = {}
    for meta in metadata:
        labels_all = meta['cells'][meta['label_column']].astype(str).to_numpy()
        rows = stratified_subset(labels_all, max_cells, seed=2026)
        labels = labels_all[rows]
        plans[meta['name']] = dict(rows=rows, labels=labels,
                                  splits={seed: split_by_type(labels, seed) for seed in seeds})
    common_labels = sorted(set.intersection(*(set(plan['labels']) for plan in plans.values())))
    if len(common_labels) < 2:
        raise ValueError('三套数据共同标签不足两类，不能完成来源匹配对照。')
    target_counts = {}
    for seed in seeds:
        target_counts[seed] = {
            label: min(int(np.sum(plan['labels'][plan['splits'][seed]['train']] == label))
                       for plan in plans.values()) for label in common_labels}
        if sum(target_counts[seed].values()) < max(K_VALUES):
            raise ValueError('匹配对照训练细胞不足40个；请提高显式抽样上限。')
    return plans, target_counts


def run_experiment(args, metadata, genes):
    global FIGURE_NOTE
    FIGURE_NOTE = (f'Pilot · {args.max_cells_per_dataset:,} cells per source'
                   if args.max_cells_per_dataset is not None else '')
    plans, target_counts = prepare_experiment(metadata, args.seeds, args.max_cells_per_dataset)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for directory in ('splits', 'predictions', 'selected_genes', 'figures'):
        (output / directory).mkdir(exist_ok=True)
    pd.Series(genes, name='gene_name').to_csv(output / 'common_genes.csv', index=False)
    manifest = {
        'status': 'running', 'profile': args.profile, 'data_dir': str(args.data_dir.resolve()),
        'seeds': args.seeds, 'label_column': args.label_column,
        'max_cells_per_dataset': args.max_cells_per_dataset,
        'sampled_run': args.max_cells_per_dataset is not None,
        'metrics': list(METRICS), 'k_values': list(K_VALUES), 'kernel_bandwidth': 0.1,
        'confidence_threshold': 0.5, 'n_common_genes': len(genes),
        'distance_scale': 'Exact maximum finite training-training distance; fixed for each metric.',
        'normalization': 'Sum duplicated gene symbols; intersect genes; per-cell total 10000 then log1p.',
        'hvg': 'Training-only mean-binned log-dispersion (20 bins), Seurat-style.',
        'umap': 'Plot only: training HVG/SVD; unsupervised UMAP of held-out test expression.',
        'matched_training_counts': target_counts,
        'python': sys.version, 'platform': platform.platform(),
        'datasets': {m['name']: {'original_cells': len(m['cells']), 'selected_cells': len(plans[m['name']]['rows']),
                                'original_genes': len(m['genes'])} for m in metadata},
        'papers': ['https://doi.org/10.1038/s41467-026-77349-8',
                   'https://doi.org/10.1186/s13059-023-03139-w'],
    }
    manifest_path = output / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    results, per_type, configuration_rows, settings = [], [], [], []
    diagnostics, evaluation_diagnostics, markers, composition = [], [], [], []
    started = time.monotonic()
    for meta in metadata:
        name = meta['name']
        plan = plans[name]
        labels = plan['labels']
        print(f'\n[{name}] 读取 {len(labels):,} 个细胞、{len(genes):,} 个共同基因', flush=True)
        counts = load_expression(meta, plan['rows'], genes, cache_dir=None)
        log_counts = log_normalize(counts)
        cells = meta['cells'].iloc[plan['rows']].reset_index(drop=True)
        for label, count in meta['cells'][args.label_column].value_counts().items():
            composition.append(dict(dataset=name, label=str(label), count=int(count)))
        for seed in args.seeds:
            for regime in ('full', 'matched'):
                split = plan['splits'][seed]
                if regime == 'matched':
                    split = matched_indices(labels, split, target_counts[seed], seed + 1000)
                train, val, test = (split[key] for key in ('train', 'val', 'test'))
                classes = np.unique(labels[train])
                if not set(labels[val]).issubset(classes) or not set(labels[test]).issubset(classes):
                    raise ValueError('验证或测试出现训练集未覆盖的标签，不能进行本闭集分类实验。')
                split_records = []
                for part, positions in split.items():
                    split_records.extend(dict(cell_id=cells.iloc[i]['cell_id'], source_row=int(plan['rows'][i]),
                                              label=labels[i], split=part) for i in positions)
                tag = f'{name}_seed{seed}_{regime}'
                pd.DataFrame(split_records).to_csv(output / 'splits' / f'{tag}.csv', index=False)
                log_train = log_counts[train]
                ranking, hvg_scores = hvg_ranking(log_train)
                selected_features = {
                    'A_raw': np.arange(len(genes)), 'B_log': np.arange(len(genes)),
                    'C_hvg2000': np.sort(ranking[:2000]), 'D_hvg4000': np.sort(ranking[:4000]),
                }
                for feature, columns in selected_features.items():
                    print(f'  seed={seed} {regime} {feature}: train/val/test={len(train)}/{len(val)}/{len(test)}, '
                          f'genes={len(columns)}', flush=True)
                    source = counts if feature == 'A_raw' else log_counts
                    x_train = source[train][:, columns].tocsr()
                    x_val = source[val][:, columns].tocsr()
                    x_test = source[test][:, columns].tocsr()
                    pd.DataFrame({'gene_name': genes[columns], 'training_hvg_score': hvg_scores[columns]}).to_csv(
                        output / 'selected_genes' / f'{tag}_{feature}.csv', index=False)
                    scales = training_distance_scales(x_train, args.block_size)
                    val_prob = knn_probabilities(x_train, labels[train], x_val, classes, scales, args.block_size)
                    selected = choose_models(val_prob, labels[val], classes)
                    context = dict(dataset=name, seed=seed, regime=regime, feature=feature,
                                   n_train=len(train), n_test=len(test), n_genes=len(columns), n_classes=len(classes))
                    for (metric, k), probability in val_prob.items():
                        summary, _ = evaluate_probabilities(probability, labels[val], classes,
                                                            dict(context, metric=metric, k=k, split='val'))
                        configuration_rows.append(summary)
                    settings.append(dict(context, scales=scales,
                                         selected={key: [list(item) for item in value] for key, value in selected.items()}))
                    test_prob = knn_probabilities(x_train, labels[train], x_test, classes, scales, args.block_size)
                    test_predictions = pd.DataFrame({'cell_id': cells.iloc[test]['cell_id'].to_numpy(),
                                                     'true_label': labels[test]})
                    for split_name, truth, config_prob in (('val', labels[val], val_prob), ('test', labels[test], test_prob)):
                        for model, keys in selected.items():
                            probability = average_probabilities(config_prob, keys)
                            summary, type_rows = evaluate_probabilities(
                                probability, truth, classes, dict(context, split=split_name, model=model))
                            results.append(summary)
                            per_type.extend(type_rows)
                            if split_name == 'test':
                                prediction = classes[probability.argmax(axis=1)]
                                test_predictions[f'{model}_label'] = prediction
                                if model == 'ensemble':
                                    test_predictions['predicted_label'] = prediction
                                    test_predictions['confidence'] = probability.max(axis=1)
                                    test_predictions['error'] = prediction != truth
                                    test_predictions['high_confidence'] = probability.max(axis=1) > 0.5
                    test_predictions.to_csv(output / 'predictions' / f'{tag}_{feature}.csv', index=False)
                    if seed == args.seeds[0] and regime == 'full' and feature == 'B_log':
                        main_diagnostic = test_predictions.copy()
                        complete_diagnostic = test_predictions.copy()
                        complete_diagnostic['dataset'] = name
                        evaluation_diagnostics.append(complete_diagnostic)
                    save_tables(output, results, per_type, configuration_rows, settings)
                    del x_train, x_val, x_test, val_prob, test_prob
                if seed == args.seeds[0] and regime == 'full':
                    markers.extend(marker_summary(log_train, labels[train], genes, name))
                    # 图只采样测试细胞；指标与预测CSV仍包含全部测试细胞。
                    plot_rows = stratified_subset(labels[test], args.plot_max_cells, seed, minimum=1)
                    coords = umap_for_test(log_train, log_counts[test[plot_rows]], ranking, seed)
                    main_diagnostic = main_diagnostic.iloc[plot_rows].copy()
                    main_diagnostic['dataset'] = name
                    main_diagnostic['umap1'], main_diagnostic['umap2'] = coords[:, 0], coords[:, 1]
                    for gene in ('Slc17a7', 'Gad1', 'Aqp4', 'Mbp'):
                        position = np.flatnonzero(genes == gene)
                        if not len(position):
                            raise ValueError(f'缺少预设marker {gene}，停止生成该图，请检查基因标识。')
                        main_diagnostic['marker_' + gene] = log_counts[test[plot_rows], int(position[0])].toarray().ravel()
                    diagnostics.append(main_diagnostic)
                del log_train
                gc.collect()
        del counts, log_counts
        gc.collect()

    diagnostic_frame = pd.concat(diagnostics, ignore_index=True)
    evaluation_frame = pd.concat(evaluation_diagnostics, ignore_index=True)
    marker_frame = pd.DataFrame(markers)
    diagnostic_frame.to_csv(output / 'plot_diagnostics.csv', index=False)
    evaluation_frame.to_csv(output / 'test_diagnostics.csv', index=False)
    marker_frame.to_csv(output / 'marker_summary.csv', index=False)
    pd.DataFrame(composition).to_csv(output / 'dataset_composition.csv', index=False)
    palette = make_label_palette(sorted({str(c) for m in metadata for c in m['cells'][args.label_column]}))
    figures = output / 'figures'
    generated = []
    generated += plot_overview(pd.DataFrame(composition), figures, palette=palette)
    generated += plot_annotation(diagnostic_frame, figures, palette=palette)
    metric_frame, per_type_frame = pd.DataFrame(results), pd.DataFrame(per_type)
    for regime in ('full', 'matched'):
        regime_dir = figures / regime
        generated += plot_performance(metric_frame, per_type_frame, regime_dir, regime=regime, palette=palette)
        generated += plot_feature_comparison(metric_frame, regime_dir, regime=regime)
        generated += plot_ablation(metric_frame, regime_dir, regime=regime)
    generated += plot_confusion(evaluation_frame, figures, palette=palette)
    generated += plot_markers(marker_frame, figures, palette=palette)
    generated += plot_confidence(evaluation_frame, figures)
    # This larger descriptive atlas is separate from the held-out predictions.
    atlas, atlas_info = build_exploratory_atlas(args.data_dir, output, args.atlas_cells, seed=args.seeds[0])
    generated += plot_exploratory_atlas(atlas, figures)
    previous_note = FIGURE_NOTE
    FIGURE_NOTE = 'Exploratory atlas'
    generated += plot_marker_expression(atlas, figures)
    FIGURE_NOTE = previous_note
    manifest['exploratory_atlas'] = {'file': 'atlas_manifest.json', 'used_for_classification': False}
    manifest.update(status='complete', elapsed_seconds=round(time.monotonic() - started, 2),
                    figure_files=[str(p.relative_to(output)) for p in generated])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成。结果：{output.resolve()}；生成 {len(generated)} 张图。', flush=True)
    if args.max_cells_per_dataset is not None:
        print('注意：此次为显式抽样验证，不代表完整数据集实验结论。', flush=True)


def _redraw_file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _redraw_preflight(source):
    """Read and validate every required saved input before any output is changed."""
    marker_columns = ['marker_' + gene for gene in ('Slc17a7', 'Gad1', 'Aqp4', 'Mbp')]
    schemas = {
        'metrics.csv': ['dataset', 'seed', 'regime', 'feature', 'model', 'split',
                        'accuracy', 'macro_f1', 'balanced_accuracy'],
        'per_class_metrics.csv': ['dataset', 'seed', 'regime', 'feature', 'model',
                                  'label', 'precision', 'recall', 'f1', 'support'],
        'plot_diagnostics.csv': ['dataset', 'cell_id', 'true_label', 'predicted_label',
                                 'confidence', 'umap1', 'umap2', *marker_columns],
        'test_diagnostics.csv': ['dataset', 'cell_id', 'true_label', 'predicted_label', 'confidence'],
        'marker_summary.csv': ['dataset', 'label', 'gene', 'mean_expression', 'fraction_expressing'],
        'dataset_composition.csv': ['dataset', 'label', 'count'],
    }
    required = ['run_manifest.json', *schemas]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f'重绘缺少必需文件，未生成结果：{", ".join(missing)}')
    manifest = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('datasets'), dict) or not manifest['datasets']:
        raise ValueError('原run_manifest.json缺少有效datasets provenance，停止重绘。')
    if 'max_cells_per_dataset' not in manifest or 'sampled_run' not in manifest:
        raise ValueError('原run_manifest.json缺少抽样设置，不能可靠标注图的来源。')
    limit = manifest['max_cells_per_dataset']
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError('原run_manifest.json的max_cells_per_dataset必须为正整数或null。')
    if bool(manifest['sampled_run']) != (limit is not None):
        raise ValueError('原run_manifest.json的抽样标记与抽样上限不一致。')
    datasets = set(manifest['datasets'])
    tables = {}
    for name, columns in schemas.items():
        frame = pd.read_csv(source / name)
        _require(frame, columns, name)
        if frame[columns].isna().any().any():
            raise ValueError(f'{name}存在缺失必需值，停止重绘，不填补数据。')
        if set(frame['dataset']) != datasets:
            raise ValueError(f'{name}的数据来源集合与run_manifest.json不一致。')
        tables[name] = frame
    metrics = tables['metrics.csv']
    per_class = tables['per_class_metrics.csv']
    for dataset in datasets:
        for regime in ('full', 'matched'):
            selected = metrics.loc[(metrics['dataset'] == dataset) & (metrics['regime'] == regime) & (metrics['split'] == 'test')]
            combinations = set(zip(selected['feature'], selected['model']))
            expected = {(feature, 'ensemble') for feature in ('A_raw', 'B_log', 'C_hvg2000', 'D_hvg4000')}
            expected |= {('B_log', model) for model in ('single', 'multi_k', 'ensemble')}
            if not expected.issubset(combinations):
                raise ValueError(f'{dataset}/{regime}缺少完整绘图所需指标：{sorted(expected - combinations)}')
            selected_types = per_class.loc[(per_class['dataset'] == dataset) & (per_class['regime'] == regime) & (per_class['feature'] == 'B_log')]
            if 'split' in selected_types:
                selected_types = selected_types.loc[selected_types['split'] == 'test']
            if not {'single', 'multi_k', 'ensemble'}.issubset(set(selected_types['model'])):
                raise ValueError(f'{dataset}/{regime}缺少B_log逐类型模型结果。')
    atlas = None
    atlas_path = source / 'atlas_coordinates.csv'
    if atlas_path.is_file():
        atlas = pd.read_csv(atlas_path)
        columns = ['dataset', 'cell_id', 'class_label', 'subclass_label', 'umap1', 'umap2', *marker_columns]
        _require(atlas, columns, 'atlas_coordinates.csv')
        if atlas[columns].isna().any().any() or set(atlas['dataset']) != datasets:
            raise ValueError('atlas_coordinates.csv含缺失值或来源不一致，停止重绘。')
        if not np.isfinite(atlas[['umap1', 'umap2', *marker_columns]].to_numpy(dtype=float)).all():
            raise ValueError('atlas_coordinates.csv含非有限坐标或表达值。')
        if (atlas[marker_columns] < 0).any().any():
            raise ValueError('atlas_coordinates.csv的实测log表达不能为负。')
    return manifest, tables, atlas


def redraw_results(args):
    """Replot saved results without inspecting raw matrices or fitting any model."""
    global FIGURE_NOTE
    source = args.redraw_from.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    manifest, tables, atlas = _redraw_preflight(source)
    copied_names = ['run_manifest.json', *tables]
    optional = ['validation_configurations.csv', 'model_settings.json', 'common_genes.csv',
                'atlas_coordinates.csv', 'atlas_manifest.json']
    copied_names += [name for name in optional if (source / name).is_file()]
    source_hashes = {name: _redraw_file_hash(source / name) for name in copied_names}
    # A repeated redraw may replace figures, but never another run's saved results.
    for name in copied_names:
        destination = output / name
        if source / name == destination:
            continue
        if destination.exists() and (not destination.is_file() or _redraw_file_hash(destination) != source_hashes[name]):
            raise FileExistsError(f'输出目录已有不同结果文件，未覆盖：{destination}')
    output.mkdir(parents=True, exist_ok=True)
    for name in copied_names:
        if source / name != output / name and not (output / name).exists():
            shutil.copy2(source / name, output / name)
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'umap-learn', 'numba'):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    polish = {
        'status': 'running', 'operation': 'redraw_saved_results',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_run': str(source), 'output_dir': str(output),
        'python_executable': sys.executable, 'python': sys.version,
        'platform': platform.platform(), 'package_versions': versions,
        'source_file_sha256': source_hashes, 'copied_source_files': copied_names,
        'original_run_manifest_preserved': True, 'models_refitted': False,
        'atlas': ({'source_file': str(source / 'atlas_coordinates.csv'),
                   'cells_per_dataset': {str(name): int(count) for name, count in atlas.groupby('dataset').size().items()},
                   'used_for': ['exploratory_atlas', 'marker_expression'],
                   'used_for_model_evaluation': False} if atlas is not None else None),
        'figure_files': [],
    }
    polish_path = output / 'polish_manifest.json'
    polish_path.write_text(json.dumps(polish, ensure_ascii=False, indent=2), encoding='utf-8')
    limit = manifest['max_cells_per_dataset']
    FIGURE_NOTE = f'Pilot · {limit:,} cells per source' if limit is not None else ''
    figures = output / 'figures'
    labels = set(tables['dataset_composition.csv']['label'].astype(str))
    for name in ('plot_diagnostics.csv', 'test_diagnostics.csv'):
        labels.update(tables[name]['true_label'].astype(str))
        labels.update(tables[name]['predicted_label'].astype(str))
    labels.update(tables['per_class_metrics.csv']['label'].astype(str))
    labels.update(tables['marker_summary.csv']['label'].astype(str))
    palette = make_label_palette(sorted(labels))
    generated = []
    try:
        generated += plot_overview(tables['dataset_composition.csv'], figures, palette=palette)
        generated += plot_annotation(tables['plot_diagnostics.csv'], figures, palette=palette)
        for regime in ('full', 'matched'):
            regime_dir = figures / regime
            generated += plot_performance(tables['metrics.csv'], tables['per_class_metrics.csv'], regime_dir, regime=regime, palette=palette)
            generated += plot_feature_comparison(tables['metrics.csv'], regime_dir, regime=regime)
            generated += plot_ablation(tables['metrics.csv'], regime_dir, regime=regime)
        generated += plot_confusion(tables['test_diagnostics.csv'], figures, palette=palette)
        generated += plot_markers(tables['marker_summary.csv'], figures, palette=palette)
        generated += plot_confidence(tables['test_diagnostics.csv'], figures)
        if atlas is not None:
            previous_note = FIGURE_NOTE
            counts = atlas.groupby('dataset').size()
            FIGURE_NOTE = (f'Exploratory · {int(counts.iloc[0]):,} cells per source'
                           if counts.nunique() == 1 else 'Exploratory atlas')
            try:
                generated += plot_exploratory_atlas(atlas, figures)
                generated += plot_marker_expression(atlas, figures)
            finally:
                FIGURE_NOTE = previous_note
        else:
            generated += plot_marker_expression(tables['plot_diagnostics.csv'], figures)
        polish.update(status='complete', figure_files=[str(Path(path).resolve().relative_to(output)) for path in generated])
    except Exception as error:
        polish.update(status='failed', error=f'{type(error).__name__}: {error}',
                      figure_files=[str(Path(path).resolve().relative_to(output)) for path in generated])
        raise
    finally:
        polish_path.write_text(json.dumps(polish, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'仅重绘完成：{output}；生成 {len(generated)} 张图。原模型结果与run_manifest.json保持不变。', flush=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Q2-1: Emerge式集成KNN细胞类型注释、特征/集成消融及论文风格图。')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data2')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--redraw-from', type=Path, help='从已有结果目录读取CSV并重绘；不读取原始矩阵、不训练模型。')
    parser.add_argument('--atlas-only', action='store_true', help='仅生成真实数据概览、探索性UMAP和marker图，不训练分类器。')
    parser.add_argument('--atlas-cells', type=int, default=1500, help='探索性图谱每来源的分层抽样上限，与分类样本数独立。')
    parser.add_argument('--label-column', choices=('class_label', 'subclass_label', 'cluster_label'), default='subclass_label')
    parser.add_argument('--profile', choices=('full', 'smoke'), default='full', help='smoke显式抽样180细胞/来源、一个seed，仍保留全部模型。')
    parser.add_argument('--seeds', type=int, nargs='+', help='正式实验默认42 43 44。')
    parser.add_argument('--max-cells-per-dataset', type=int, help='显式的开发用分层抽样上限；默认使用完整数据。')
    parser.add_argument('--plot-max-cells', type=int, default=5000, help='仅限制UMAP绘图点数，不限制评价集。')
    parser.add_argument('--block-size', type=int, default=128, help='精确距离计算的行分块大小。')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-only', action='store_true', help='仅核对必需文件、CSV/NPZ结构、共同基因和标签，不跑模型。')
    args = parser.parse_args()
    if sum([args.redraw_from is not None, args.check_only, args.atlas_only]) > 1:
        parser.error('--redraw-from、--check-only与--atlas-only不能同时使用。')
    if args.seeds is None:
        args.seeds = [42] if args.profile == 'smoke' else [42, 43, 44]
    if args.profile == 'smoke' and args.max_cells_per_dataset is None:
        args.max_cells_per_dataset = 180
    if args.output_dir is None:
        if args.redraw_from is not None:
            source = args.redraw_from.expanduser().resolve()
            args.output_dir = source.parent / (source.name + '_polished')
        else:
            suffix = '_atlas' if args.atlas_only else ('_smoke_results' if args.profile == 'smoke' else '_results')
            args.output_dir = Path(__file__).resolve().parent / ('Q2-1' + suffix)
    if args.block_size < 1 or args.threads < 1 or args.plot_max_cells < 4:
        parser.error('block-size、threads必须为正数，plot-max-cells至少为4。')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds不能重复。')
    if args.atlas_cells < 60:
        parser.error('atlas-cells至少60，以保留各细胞亚类。')
    return args


def main():
    global FIGURE_NOTE
    args = parse_arguments()
    if args.redraw_from is not None:
        redraw_results(args)
        return
    if args.atlas_only:
        atlas, atlas_info = build_exploratory_atlas(args.data_dir, args.output_dir, args.atlas_cells, seed=args.seeds[0])
        composition = []
        for name in DATASET_NAMES:
            meta = inspect_dataset(args.data_dir, name)
            composition.extend({'dataset': name, 'label': label, 'count': int(count)}
                               for label, count in meta['cells']['subclass_label'].value_counts().items())
        frame = pd.DataFrame(composition)
        frame.to_csv(args.output_dir / 'dataset_composition.csv', index=False)
        FIGURE_NOTE = 'Exploratory atlas'
        figures = args.output_dir / 'figures'
        plot_overview(frame, figures)
        plot_exploratory_atlas(atlas, figures)
        plot_marker_expression(atlas, figures)
        print(f'探索性图谱完成：{args.output_dir.resolve()}；没有训练或评价分类器。', flush=True)
        return
    metadata = [inspect_dataset(args.data_dir, name, args.label_column) for name in DATASET_NAMES]
    genes = common_gene_names(metadata)
    if len(genes) < 4000:
        raise ValueError(f'只有 {len(genes)} 个共同基因，不能按约定执行4000-HVG方案。')
    # 所有必需输入先核对完毕；数据缺失或结构错误立即停止，不跳过任何来源。
    for meta in metadata:
        print(f"{meta['name']}: cells={len(meta['cells']):,}, genes={len(meta['genes']):,}, "
              f"labels={meta['cells'][args.label_column].nunique()}", flush=True)
    print(f'共同基因：{len(genes):,}；按gene_name对齐，同名列求和。', flush=True)
    prepare_experiment(metadata, args.seeds, args.max_cells_per_dataset)
    if args.check_only:
        print('文件、矩阵头和元数据检查通过；表达值的完整校验将在流式读取时执行。')
        return
    _configure_atlas_environment(args.output_dir)
    try:
        from umap import UMAP  # 提前检查绘图依赖，避免计算完模型才发现缺包。
    except ImportError as error:
        raise ImportError('缺少 umap-learn。请运行 python -m pip install umap-learn，再启动本实验。') from error
    print('精确五距离全基因KNN计算量很大。首次建议 --profile smoke；不会自动改用近似近邻。', flush=True)
    try:
        with threadpool_limits(limits=args.threads):
            run_experiment(args, metadata, genes)
    except Exception as error:
        manifest_path = args.output_dir / 'run_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
