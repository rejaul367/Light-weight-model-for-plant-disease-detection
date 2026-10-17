"""
make_svg.py -- SVG copies of every figure and table made by section E.

It reads the results that your finished runs already saved in outputs/
(metrics.json, history.json, preds.npz) and draws the same images again with the
same functions from smg_lab.py, but saves them as vector SVG.

  * No training, no GPU.
  * Your existing PNG files, CSV files and the Word report are NOT changed.
  * Nothing in smg_lab.py or the notebook is changed.

Output:
    outputs/svg/figures/*.svg   (metrics_lines, curves, confusion, roc)
    outputs/svg/tables/*.svg    (table_<dataset>, ablation_<dataset>)
    outputs/svg_images.zip      (all SVG files in one zip, easy to download)
On Colab everything is also copied to MyDrive/SMG_Lite/outputs/.

Run it from the notebook (after the setup + CONFIG + 0.2 cells), in ONE cell:
    exec(open(os.path.join(DRIVE_DIR if IN_COLAB else ".", "make_svg.py")).read())
"""


def _smg_make_svg():
    import os, sys, glob, zipfile
    import matplotlib
    import smg_lab as lab

    if not lab.CFG.get("datasets"):
        raise RuntimeError("smg_lab is not configured yet -> run the setup cell, the CONFIG "
                           "cell and cell 0.2 (extra_models) first, then run this cell again.")

    out = lab.CFG["out_dir"]
    svg_dir = os.path.join(out, "svg")
    fdir = os.path.join(svg_dir, "figures")
    tdir = os.path.join(svg_dir, "tables")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)

    # dataset order: TABLE_ORDER from the notebook if it exists, then any other dataset with results
    g = globals()
    order = list(g.get("TABLE_ORDER") or [])
    order += [d for d in lab.CFG["datasets"] if d not in order]
    table_sets = [d for d in order if os.path.isdir(os.path.join(out, "runs", d))]
    ablation_sets = [d for d in order if os.path.isdir(os.path.join(out, "ablation", d))]

    saved, skipped = [], []

    def keep(path):
        if path and os.path.exists(path):
            saved.append(path)
        return path

    # vector output; text drawn as shapes so it looks identical on every computer
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "smg_lite"}):
        for ds in table_sets:
            df, res = lab.results_table(ds)
            keep(lab.table_png(df, res, os.path.join(tdir, f"table_{ds}.svg")))
            for fn, name in ((lab.plot_metric_lines, "metrics_lines"), (lab.plot_curves, "curves"),
                             (lab.plot_confusion, "confusion"), (lab.plot_roc, "roc")):
                path = os.path.join(fdir, f"{name}_{ds}.svg")
                if keep(fn(ds, path)) is None:
                    skipped.append(f"{name}_{ds} (results not found)")
        for ds in ablation_sets:
            df, res = lab.ablation_table(ds)
            keep(lab.table_png(df, res, os.path.join(tdir, f"ablation_{ds}.svg"), check_cols=3))

    # plot_confusion also writes a class-index CSV next to its image -> remove that duplicate
    # here (the original CSV in outputs/figures/ is untouched)
    for f in glob.glob(os.path.join(fdir, "*_class_index.csv")):
        os.remove(f)

    zpath = os.path.join(out, "svg_images.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in saved:
            z.write(f, os.path.relpath(f, svg_dir))

    if lab.CFG.get("backup_dir"):          # Colab: copy to MyDrive/SMG_Lite/outputs/
        lab._backup(svg_dir)
        lab._backup(zpath)

    print(f"{len(saved)} SVG file(s) written:")
    for f in saved:
        print("  ", f)
    for s in skipped:
        print("   skipped:", s)
    print(f"\nall SVG in one zip -> {zpath}")
    if lab.CFG.get("backup_dir"):
        print(f"also saved in       -> {os.path.join(lab.CFG['backup_dir'], 'svg')}")
    print("PNG files in outputs/figures and outputs/tables were not changed.")
    return saved


_smg_make_svg()
