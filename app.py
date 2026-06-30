"""
app.py  –  Raman CNN Dashboard  (Streamlit)
============================================
Run:  streamlit run app.py

Tabs
----
1. Training Monitor   – live loss/accuracy curves from training_log.csv
2. Model Info         – architecture, class distribution
3. Evaluation         – confusion matrix, classification report
4. Saliency Explorer  – per-class Integrated-Gradient maps
5. Live Inference     – classify a new spectrum from the database
"""

import os, json, time, glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raman CNN Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT      = "outputs"
LOG_CSV  = os.path.join(OUT, "logs",  "training_log.csv")
REG_CSV  = os.path.join(OUT, "logs",  "key_spectral_regions.csv")
SAL_NPZ  = os.path.join(OUT, "logs",  "saliency_maps.npz")
CFG_JSON = os.path.join(OUT, "model", "model_config.json")
BEST_PT  = os.path.join(OUT, "model", "best_model.pt")
HYBRID_ARTIFACT = os.path.join(OUT, "model", "hybrid", "hybrid_ensemble.joblib")
CM_PNG   = os.path.join(OUT, "figures", "confusion_matrix.png")
TC_PNG   = os.path.join(OUT, "figures", "training_curves.png")
CD_PNG   = os.path.join(OUT, "figures", "class_distribution.png")
CD_SPLIT_PNG = os.path.join(OUT, "figures", "class_distribution_splits.png")
CLASS_PROTO_PNG = os.path.join(OUT, "figures", "class_prototypes.png")
CONFUSED_PROTO_PNG = os.path.join(OUT, "figures", "confused_pair_prototypes.png")
CLASS_COUNTS_CSV = os.path.join(OUT, "logs", "class_counts.csv")
PAIR_ANALYSIS_CSV = os.path.join(OUT, "logs", "confusion_pair_analysis.csv")
PER_CLASS_CSV = os.path.join(OUT, "logs", "per_class_metrics.csv")
TOP_CONFUSIONS_CSV = os.path.join(OUT, "logs", "top_confusions.csv")
WRONG_PRED_CSV = os.path.join(OUT, "logs", "wrong_predictions.csv")
MIX_CFG_JSON = os.path.join(OUT, "model", "model_config_mixture.json")
MIX_BEST_PT  = os.path.join(OUT, "model", "best_model_mixture.pt")
MIX_LOG_CSV  = os.path.join(OUT, "logs", "training_log_mixture.csv")
MIX_CM_PNG   = os.path.join(OUT, "figures", "confusion_matrix_mixture.png")
MIX_CM_CSV   = os.path.join(OUT, "logs", "confusion_matrix_mixture.csv")
MIX_SAL_NPZ  = os.path.join(OUT, "logs", "saliency_maps_mixture.npz")
MIX_REG_CSV  = os.path.join(OUT, "logs", "key_spectral_regions_mixture.csv")
MIX_SAL_HM_PNG = os.path.join(OUT, "figures", "saliency_heatmap_all_mixture.png")

META_CSV    = "ramanbiolib/db/metadata_db.csv"
SPECTRA_CSV = "ramanbiolib/db/raman_spectra_db.csv"

# ── Helper: parse list strings from CSV ───────────────────────────────────────
def parse_list(s):
    return [float(v) for v in str(s).strip("[]").split(", ") if v]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def format_metric(value, precision=1):
    return f"{value:.{precision}%}" if isinstance(value, (int, float)) else "N/A"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Raman CNN Dashboard")
st.sidebar.markdown("**ramanbiolib** — 1D CNN + Integrated Gradients")
st.sidebar.divider()

# Training status badge — prefer the production hybrid ensemble's metrics;
# fall back to the single-CNN model_config.json only if the hybrid isn't built.
HYBRID_METRICS = os.path.join(OUT, "model", "hybrid", "metrics.json")
training_done = os.path.exists(LOG_CSV) or os.path.exists(HYBRID_METRICS)
if training_done:
    if os.path.exists(HYBRID_METRICS):
        hm = json.load(open(HYBRID_METRICS))
        hold = hm.get("holdout_test_acc")
        cv = hm.get("cv_acc")
        parts = []
        if isinstance(hold, (int, float)):
            parts.append(f"Holdout acc: **{hold:.1%}**")
        if isinstance(cv, (int, float)):
            parts.append(f"CV acc: **~{cv:.0%}**")
        st.sidebar.success("✅ Hybrid ensemble ready\n" + "  ·  ".join(parts or ["ready"]))
    else:
        cfg = json.load(open(CFG_JSON)) if os.path.exists(CFG_JSON) else {}
        test_acc = cfg.get('test_acc', None)
        acc_str = f"{test_acc:.1%}" if isinstance(test_acc, (int, float)) else "N/A"
        st.sidebar.success(f"✅ Training complete (single CNN)\nTest acc: **{acc_str}**")
else:
    st.sidebar.warning("⏳ Training in progress…")
    st.sidebar.caption("Dashboard auto-refreshes every 10 s")

if os.path.exists(BEST_PT):
    size_mb = os.path.getsize(BEST_PT) / 1e6
    st.sidebar.info(f"💾 best_model.pt  ({size_mb:.1f} MB)")

st.sidebar.divider()
tab_names = ["📈 Training Monitor", "🏗️ Model Info",
             "📊 Evaluation", "🌡️ Saliency Explorer", "⚡ Live Inference",
             "🧪 Mixture Evaluation", "🧪 Mixture Saliency", "🧪 Mixture Inference"]
selected = st.sidebar.radio("Navigate", tab_names)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Monitor
# ═══════════════════════════════════════════════════════════════════════════════
if selected == tab_names[0]:
    st.title("📈 Training Monitor")

    if not os.path.exists(LOG_CSV):
        # Show partial progress from best_model.pt existence
        st.info("Training is running in the background. This page auto-refreshes every 10 seconds.")
        col1, col2 = st.columns(2)
        col1.metric("Model checkpoint", "✅ saved" if os.path.exists(BEST_PT) else "⏳ waiting")
        col2.metric("Class distribution", "✅ saved" if os.path.exists(CD_PNG) else "⏳ waiting")
        if os.path.exists(CD_PNG):
            st.image(CD_PNG, caption="Class Distribution", use_container_width=True)
        st.info("Training log will appear here once training completes.")
        time.sleep(10)
        st.rerun()
    else:
        df = pd.read_csv(LOG_CSV)
        epochs = list(range(1, len(df) + 1))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Cross-Entropy Loss", "Classification Accuracy"])
        fig.add_trace(go.Scatter(x=epochs, y=df["train_loss"], name="Train loss", line=dict(color="#4C8BF5")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=df["val_loss"],   name="Val loss",   line=dict(color="#F5A623")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=df["train_acc"],  name="Train acc",  line=dict(color="#4C8BF5")), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=df["val_acc"],    name="Val acc",    line=dict(color="#F5A623")), row=1, col=2)
        fig.update_xaxes(title_text="Epoch")
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final train loss", f"{df['train_loss'].iloc[-1]:.4f}")
        col2.metric("Final val loss",   f"{df['val_loss'].iloc[-1]:.4f}")
        col3.metric("Best train acc",   f"{df['train_acc'].max():.3f}")
        col4.metric("Best val acc",     f"{df['val_acc'].max():.3f}")

        st.dataframe(df.round(4), use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[1]:
    st.title("🏗️ Model & Data Info")

    # Class distribution
    if os.path.exists(CD_PNG):
        st.subheader("Class Distribution")
        st.image(CD_PNG, use_container_width=True)
    if os.path.exists(CD_SPLIT_PNG):
        st.subheader("Class Counts by Split")
        st.image(CD_SPLIT_PNG, use_container_width=True)
    if os.path.exists(CLASS_COUNTS_CSV):
        df_counts = pd.read_csv(CLASS_COUNTS_CSV)
        st.subheader("Split Count Audit")
        st.dataframe(df_counts, use_container_width=True, height=260)
    if os.path.exists(CLASS_PROTO_PNG):
        st.subheader("Class Prototype Spectra")
        st.image(CLASS_PROTO_PNG, use_container_width=True)

    # Dataset stats
    try:
        meta = pd.read_csv(META_CSV)
        spec = pd.read_csv(SPECTRA_CSV,
                           converters={"wavenumbers": parse_list, "intensity": parse_list})
        merged = spec.merge(meta[["id","type"]].drop_duplicates("id"), on="id")
        merged["class"] = merged["type"].str.split("/").str[0]
        KEEP = ["Proteins","Lipids","Saccharides","AminoAcids","PrimaryMetabolites","NucleicAcids"]
        merged = merged[merged["class"].isin(KEEP)]
        counts = merged["class"].value_counts().reset_index()
        counts.columns = ["Class","Spectra"]

        fig = px.bar(counts, x="Class", y="Spectra", color="Class",
                     title="Spectra per class (ramanbiolib)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        wn = np.array(merged["wavenumbers"].iloc[0])
        col1, col2, col3 = st.columns(3)
        col1.metric("Total spectra (6 classes)", len(merged))
        col2.metric("Wavenumber points", len(wn))
        col3.metric("Wavenumber range", f"{wn[0]:.0f}–{wn[-1]:.0f} cm⁻¹")
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

    # Architecture
    st.subheader("CNN Architecture")
    st.code("""
RamanCNN1D  (input: batch × 1 × 1351)
│
├─ Block 1: Conv1D(1→32, k=15) ×2  + BN + ReLU + MaxPool(4) + Dropout(0.25)
├─ Block 2: Conv1D(32→64, k=11) ×2 + BN + ReLU + MaxPool(4) + Dropout(0.25)
├─ Block 3: Conv1D(64→128, k=7)    + BN + ReLU + MaxPool(4) + Dropout(0.25)
│
├─ Flatten → Linear(→256) + ReLU + Dropout(0.4)
└─ Linear(256 → 6 classes)

Parameters : 831,654
Optimizer  : Adam  (lr=1e-3, wd=1e-4)
Scheduler  : CosineAnnealingLR (T_max=80)
Batch size : 32  (WeightedRandomSampler for class balance)
Augment    : ×15 Gaussian noise (σ=0.015) + amplitude scaling (0.85–1.15)
""", language="text")

    if os.path.exists(CFG_JSON):
        cfg = json.load(open(CFG_JSON))
        st.json(cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[2]:
    st.title("📊 Evaluation")

    if not training_done:
        st.info("⏳ Waiting for training to complete…")
        time.sleep(10); st.rerun()
    else:
        if os.path.exists(CM_PNG):
            st.subheader("Confusion Matrix")
            st.image(CM_PNG, use_container_width=True)

        if os.path.exists(TC_PNG):
            st.subheader("Training Curves")
            st.image(TC_PNG, use_container_width=True)

        if os.path.exists(REG_CSV):
            st.subheader("Top Spectral Regions per Class (from Integrated Gradients)")
            df_reg = pd.read_csv(REG_CSV)
            st.dataframe(df_reg, use_container_width=True)

        if os.path.exists(PER_CLASS_CSV):
            st.subheader("Per-class Metrics")
            df_metrics = pd.read_csv(PER_CLASS_CSV)
            st.dataframe(df_metrics, use_container_width=True)

        if os.path.exists(TOP_CONFUSIONS_CSV):
            st.subheader("Top Confusions")
            df_conf = pd.read_csv(TOP_CONFUSIONS_CSV)
            st.dataframe(df_conf.head(20), use_container_width=True)
            if os.path.exists(CONFUSED_PROTO_PNG):
                st.image(CONFUSED_PROTO_PNG, use_container_width=True)

        if os.path.exists(PAIR_ANALYSIS_CSV):
            st.subheader("Targeted Pair Analysis")
            df_pair = pd.read_csv(PAIR_ANALYSIS_CSV)
            st.dataframe(df_pair, use_container_width=True)
            pair_imgs = sorted(glob.glob(os.path.join(OUT, "figures", "confusion_pair_*_to_*.png")))
            for img_path in pair_imgs:
                st.image(img_path, use_container_width=True)

        if os.path.exists(WRONG_PRED_CSV):
            st.subheader("Wrong Predictions")
            df_wrong = pd.read_csv(WRONG_PRED_CSV)
            st.dataframe(df_wrong, use_container_width=True, height=260)
            wrong_pages = sorted(glob.glob(os.path.join(OUT, "figures", "wrong_predictions_page_*.png")))
            for img_path in wrong_pages:
                st.image(img_path, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Saliency Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[3]:
    st.title("🌡️ Saliency Explorer (Integrated Gradients)")

    if not os.path.exists(SAL_NPZ):
        st.info("⏳ Saliency maps not yet generated. Waiting for training to complete…")
        time.sleep(10); st.rerun()
    else:
        data = np.load(SAL_NPZ, allow_pickle=True)
        wn   = data["wavenumbers"]
        classes = list(data["class_names"])

        sel_class = st.selectbox("Select molecular class", classes)
        sal = data[sel_class]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

        # Load mean spectrum
        try:
            spec_df = pd.read_csv(SPECTRA_CSV,
                                  converters={"wavenumbers": parse_list, "intensity": parse_list})
            meta_df = pd.read_csv(META_CSV)
            merged  = spec_df.merge(meta_df[["id","type"]].drop_duplicates("id"), on="id")
            merged["class"] = merged["type"].str.split("/").str[0]
            X_cls = np.array(merged[merged["class"]==sel_class]["intensity"].tolist())
            mean_spec = X_cls.mean(axis=0) if len(X_cls) > 0 else np.zeros_like(wn)
        except Exception:
            mean_spec = np.zeros_like(wn)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=wn, y=mean_spec, name="Mean spectrum",
                                 line=dict(color="#4C8BF5", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=wn, y=sal_n, name="|IG| saliency",
                                 fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
                                 line=dict(color="crimson", width=1)), secondary_y=True)
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)")
        fig.update_yaxes(title_text="Intensity", secondary_y=False)
        fig.update_yaxes(title_text="Normalised |IG|", secondary_y=True)
        fig.update_layout(title=f"Saliency Map — {sel_class}", height=450,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap of all classes
        st.subheader("All-class saliency heatmap")
        sal_matrix = np.array([(data[c]-data[c].min())/(data[c].max()-data[c].min()+1e-9) for c in classes])
        step = 5
        fig2 = px.imshow(sal_matrix[:, ::step],
                         x=[f"{v:.0f}" for v in wn[::step]],
                         y=classes,
                         color_continuous_scale="hot",
                         aspect="auto",
                         title="Integrated-Gradient Saliency (all classes)",
                         labels={"x": "Wavenumber (cm⁻¹)", "color": "|IG|"})
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        if os.path.exists(REG_CSV):
            df_reg = pd.read_csv(REG_CSV)
            st.subheader(f"Top wavenumber regions — {sel_class}")
            st.dataframe(df_reg[df_reg["class"]==sel_class].reset_index(drop=True),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Inference
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[4]:
    st.title("⚡ Live Inference")

    if not os.path.exists(BEST_PT):
        st.info("⏳ Model checkpoint not yet available.")
        time.sleep(5); st.rerun()

    try:
        import torch, torch.nn as nn

        class RamanCNN1D(nn.Module):
            def __init__(
                self,
                input_len=1351,
                n_classes=6,
                cnn_channels=(24, 48, 96),
                classifier_hidden=128,
                conv_dropout=0.15,
                dense_dropout=0.15,
            ):
                super().__init__()
                c1, c2, c3 = cnn_channels
                self.block1 = nn.Sequential(
                    nn.Conv1d(1, c1, 15, padding=7), nn.BatchNorm1d(c1), nn.ReLU(),
                    nn.Conv1d(c1, c1, 15, padding=7), nn.BatchNorm1d(c1), nn.ReLU(),
                    nn.MaxPool1d(4), nn.Dropout(conv_dropout))
                self.block2 = nn.Sequential(
                    nn.Conv1d(c1, c2, 11, padding=5), nn.BatchNorm1d(c2), nn.ReLU(),
                    nn.Conv1d(c2, c2, 11, padding=5), nn.BatchNorm1d(c2), nn.ReLU(),
                    nn.MaxPool1d(4), nn.Dropout(conv_dropout))
                self.block3 = nn.Sequential(
                    nn.Conv1d(c2, c3, 7, padding=3), nn.BatchNorm1d(c3), nn.ReLU(),
                    nn.MaxPool1d(4), nn.Dropout(conv_dropout))
                dummy = torch.zeros(1,1,input_len)
                flat = self._fwd(dummy).shape[1]
                self.classifier = nn.Sequential(
                    nn.Linear(flat, classifier_hidden), nn.ReLU(), nn.Dropout(dense_dropout),
                    nn.Linear(classifier_hidden, n_classes))
            def _fwd(self,x):
                return self.block3(self.block2(self.block1(x))).view(x.size(0),-1)
            def forward(self,x):
                return self.classifier(self._fwd(x))

        @st.cache_resource
        def load_model():
            cfg = json.load(open(CFG_JSON)) if os.path.exists(CFG_JSON) else {
                "input_len": 1351,
                "n_classes": 6,
                "class_names": ["AminoAcids", "Lipids", "NucleicAcids", "PrimaryMetabolites", "Proteins", "Saccharides"],
                "cnn_channels": [24, 48, 96],
                "classifier_hidden": 128,
                "conv_dropout": 0.15,
                "dense_dropout": 0.15,
            }
            m = RamanCNN1D(
                cfg["input_len"],
                cfg["n_classes"],
                cnn_channels=tuple(cfg.get("cnn_channels", [24, 48, 96])),
                classifier_hidden=int(cfg.get("classifier_hidden", 128)),
                conv_dropout=float(cfg.get("conv_dropout", 0.15)),
                dense_dropout=float(cfg.get("dense_dropout", 0.15)),
            )
            m.load_state_dict(torch.load(BEST_PT, map_location="cpu"))
            m.eval()
            return m, cfg["class_names"]

        model, CLASS_NAMES = load_model()

        # ── Optional high-accuracy hybrid ensemble (classical soft-vote + TTA) ──
        # Built by build_hybrid_artifact.py; ~96% holdout / ~82% CV on data/merged,
        # versus the single CNN. Loaded if present; otherwise we fall back to the CNN.
        @st.cache_resource
        def load_hybrid():
            if not os.path.exists(HYBRID_ARTIFACT):
                return None
            import joblib
            return joblib.load(HYBRID_ARTIFACT)

        def _shift_pad(x, s):
            if s == 0:
                return x.copy()
            o = np.empty_like(x)
            if s > 0:
                o[:s] = x[0]; o[s:] = x[:-s]
            else:
                k = -s; o[-k:] = x[-1]; o[:-k] = x[k:]
            return o

        def hybrid_predict(art, intensity):
            """Soft-vote of the kept classical models with test-time augmentation.
            Returns (probs over art['class_names'])."""
            shifts = art.get("tta_shifts", [0])
            ens = None
            for name in art["kept"]:
                mdl = art["models"][name]
                acc = None
                for sh in shifts:
                    xs = _shift_pad(intensity.astype(np.float32), int(sh)).reshape(1, -1)
                    p = mdl.predict_proba(xs)[0]
                    acc = p if acc is None else acc + p
                p = acc / len(shifts)
                ens = art["weights"][name] * p if ens is None else ens + art["weights"][name] * p
            return ens

        hybrid_art = load_hybrid()
        use_hybrid = st.checkbox(
            "Use high-accuracy hybrid ensemble (recommended)",
            value=hybrid_art is not None,
            disabled=hybrid_art is None,
            help=("Classical soft-vote ensemble + test-time augmentation. "
                  "Reaches ~96% on the held-out split vs the single CNN. "
                  "Run build_hybrid_artifact.py to (re)build it.")
            if hybrid_art is not None else
            "Artifact not found — run: python build_hybrid_artifact.py",
        )

        # Load DB spectra
        spec_df = pd.read_csv(SPECTRA_CSV,
                              converters={"wavenumbers":parse_list,"intensity":parse_list})
        meta_df = pd.read_csv(META_CSV)
        merged = spec_df.merge(meta_df[["id","type"]].drop_duplicates("id"), on="id")
        merged["class"] = merged["type"].str.split("/").str[0]
        KEEP = ["Proteins","Lipids","Saccharides","AminoAcids","PrimaryMetabolites","NucleicAcids"]
        merged = merged[merged["class"].isin(KEEP)]

        st.subheader("Select a spectrum from the database")
        component_list = merged["component"].tolist()
        selected_comp = st.selectbox("Component", component_list)
        row = merged[merged["component"] == selected_comp].iloc[0]
        intensity = np.array(row["intensity"], dtype=np.float32)
        wn = np.array(row["wavenumbers"])
        true_class = row["class"]

        # Plot spectrum
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wn, y=intensity, mode="lines", name=selected_comp,
                                 line=dict(color="#4C8BF5", width=1.5)))
        fig.update_layout(title=f"Spectrum: {selected_comp}",
                          xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity",
                          height=350)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🔍 Run Inference", type="primary"):
            if use_hybrid and hybrid_art is not None:
                probs = hybrid_predict(hybrid_art, intensity)
                names = hybrid_art["class_names"]
                pred_idx = int(probs.argmax())
                pred_class = names[pred_idx]
                model_label = "Hybrid ensemble + TTA"
            else:
                names = CLASS_NAMES
                x = torch.tensor(intensity).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                    probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                pred_idx = int(probs.argmax())
                pred_class = names[pred_idx]
                model_label = "Single CNN"
            st.caption(f"Classifier: **{model_label}**")

            col1, col2 = st.columns(2)
            match = pred_class == true_class
            col1.metric("Predicted class", pred_class,
                        delta="✓ correct" if match else f"✗ true: {true_class}",
                        delta_color="normal" if match else "inverse")
            col2.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")

            fig_prob = px.bar(
                x=names, y=probs*100,
                labels={"x":"Class","y":"Probability (%)"},
                title="Class probabilities",
                color=names,
                height=300
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.info("Ensure the model checkpoint (best_model.pt) exists and training has completed.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Mixture Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[5]:
    st.title("🧪 Mixture Evaluation")
    st.caption("Artifacts from: python train_cnn_raman.py --task mixture")

    if not os.path.exists(MIX_CFG_JSON):
        st.info("Mixture model config not found yet. Train with: python train_cnn_raman.py --task mixture")
    else:
        cfg = json.load(open(MIX_CFG_JSON))
        st.metric("Mixture test macro-F1", f"{cfg.get('test_macro_f1', 'N/A')}")

        if os.path.exists(MIX_LOG_CSV):
            dfm = pd.read_csv(MIX_LOG_CSV)
            epochs = list(range(1, len(dfm) + 1))
            figm = make_subplots(rows=1, cols=2, subplot_titles=["BCE Loss", "Macro-F1"])
            figm.add_trace(go.Scatter(x=epochs, y=dfm["train_loss"], name="Train loss", line=dict(color="#4C8BF5")), row=1, col=1)
            figm.add_trace(go.Scatter(x=epochs, y=dfm["val_loss"], name="Val loss", line=dict(color="#F5A623")), row=1, col=1)
            figm.add_trace(go.Scatter(x=epochs, y=dfm["train_f1"], name="Train macro-F1", line=dict(color="#4C8BF5")), row=1, col=2)
            figm.add_trace(go.Scatter(x=epochs, y=dfm["val_f1"], name="Val macro-F1", line=dict(color="#F5A623")), row=1, col=2)
            figm.update_xaxes(title_text="Epoch")
            figm.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(figm, use_container_width=True)

        if os.path.exists(MIX_CM_PNG):
            st.subheader("Mixture confusion matrix (one-vs-rest per class)")
            st.image(MIX_CM_PNG, use_container_width=True)
        if os.path.exists(MIX_CM_CSV):
            st.subheader("Mixture confusion counts")
            st.dataframe(pd.read_csv(MIX_CM_CSV), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Mixture Saliency Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[6]:
    st.title("🧪 Mixture Saliency Explorer")
    st.caption("Separate from the original saliency explorer")

    if not os.path.exists(MIX_SAL_NPZ):
        st.info("Mixture saliency maps not found yet. Train with: python train_cnn_raman.py --task mixture")
    else:
        mix_data = np.load(MIX_SAL_NPZ, allow_pickle=True)
        wn = mix_data["wavenumbers"]
        classes = list(mix_data["class_names"])

        sel_class = st.selectbox("Select mixture class", classes)
        sal = mix_data[sel_class]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wn, y=sal_n, name="Mixture |IG| saliency",
            fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
            line=dict(color="crimson", width=1)
        ))
        fig.update_layout(
            title=f"Mixture Saliency — {sel_class}",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Normalised |IG|",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All-class mixture saliency heatmap")
        sal_matrix = np.array([
            (mix_data[c] - mix_data[c].min()) / (mix_data[c].max() - mix_data[c].min() + 1e-9)
            for c in classes
        ])
        step = 5
        fig2 = px.imshow(
            sal_matrix[:, ::step],
            x=[f"{v:.0f}" for v in wn[::step]],
            y=classes,
            color_continuous_scale="hot",
            aspect="auto",
            title="Integrated-Gradient Saliency (mixture model)",
            labels={"x": "Wavenumber (cm⁻¹)", "color": "|IG|"}
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        if os.path.exists(MIX_SAL_HM_PNG):
            st.image(MIX_SAL_HM_PNG, caption="Saved mixture saliency heatmap", use_container_width=True)
        if os.path.exists(MIX_REG_CSV):
            df_mix_reg = pd.read_csv(MIX_REG_CSV)
            st.subheader(f"Top mixture regions — {sel_class}")
            st.dataframe(
                df_mix_reg[df_mix_reg["class"] == sel_class].reset_index(drop=True),
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Mixture Inference (multi-label)
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[7]:
    st.title("🧪 Mixture Inference (Multi-label)")
    st.caption("Requires artifacts from: python train_cnn_raman.py --task mixture")

    if not os.path.exists(MIX_BEST_PT) or not os.path.exists(MIX_CFG_JSON):
        st.info("Mixture model not found yet. Train with: python train_cnn_raman.py --task mixture")
    else:
        try:
            import torch, torch.nn as nn

            class RamanCNN1D(nn.Module):
                def __init__(self, input_len=1351, n_classes=6):
                    super().__init__()
                    self.block1 = nn.Sequential(
                        nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.Conv1d(32, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.MaxPool1d(4), nn.Dropout(0.25))
                    self.block2 = nn.Sequential(
                        nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.Conv1d(64, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.MaxPool1d(4), nn.Dropout(0.25))
                    self.block3 = nn.Sequential(
                        nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
                        nn.MaxPool1d(4), nn.Dropout(0.25))
                    dummy = torch.zeros(1, 1, input_len)
                    flat = self._fwd(dummy).shape[1]
                    self.classifier = nn.Sequential(
                        nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
                        nn.Linear(256, n_classes))
                def _fwd(self, x):
                    return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)
                def forward(self, x):
                    return self.classifier(self._fwd(x))

            @st.cache_resource
            def load_mixture_model():
                cfg = json.load(open(MIX_CFG_JSON))
                model = RamanCNN1D(cfg["input_len"], cfg["n_classes"])
                model.load_state_dict(torch.load(MIX_BEST_PT, map_location="cpu"))
                model.eval()
                return model, cfg

            mix_model, mix_cfg = load_mixture_model()
            class_names = mix_cfg["class_names"]

            spec_df = pd.read_csv(
                SPECTRA_CSV,
                converters={"wavenumbers": parse_list, "intensity": parse_list}
            )
            meta_df = pd.read_csv(META_CSV)
            merged = spec_df.merge(meta_df[["id", "type"]].drop_duplicates("id"), on="id")
            merged["class"] = merged["type"].str.split("/").str[0]
            merged = merged[merged["class"].isin(class_names)].reset_index(drop=True)

            st.subheader("Build a synthetic mixture from existing spectra")
            n_comp = st.slider("Number of components", min_value=2, max_value=4, value=2, step=1)
            threshold = st.slider("Prediction threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

            selected_rows = []
            weights = []
            cols = st.columns(n_comp)
            for i in range(n_comp):
                with cols[i]:
                    cls = st.selectbox(f"Class {i+1}", options=class_names, index=i % len(class_names), key=f"mix_cls_{i}")
                    subset = merged[merged["class"] == cls]
                    comp = st.selectbox(f"Component {i+1}", options=subset["component"].tolist(), key=f"mix_comp_{i}")
                    wt = st.number_input(f"Weight {i+1}", min_value=0.0, value=float(1.0 / n_comp), step=0.05, key=f"mix_wt_{i}")
                    row = subset[subset["component"] == comp].iloc[0]
                    selected_rows.append(row)
                    weights.append(float(wt))

            w = np.array(weights, dtype=np.float32)
            w = np.ones_like(w) / len(w) if w.sum() <= 0 else (w / w.sum())
            wn = np.array(selected_rows[0]["wavenumbers"], dtype=np.float32)
            mix_intensity = np.zeros_like(np.array(selected_rows[0]["intensity"], dtype=np.float32))
            true_labels = set()
            for wi, row in zip(w, selected_rows):
                mix_intensity += wi * np.array(row["intensity"], dtype=np.float32)
                true_labels.add(row["class"])

            fig_mix = go.Figure()
            fig_mix.add_trace(go.Scatter(x=wn, y=mix_intensity, mode="lines", name="Synthetic mixture",
                                         line=dict(color="#4C8BF5", width=1.5)))
            fig_mix.update_layout(
                title="Synthetic mixture spectrum",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Intensity",
                height=320
            )
            st.plotly_chart(fig_mix, use_container_width=True)

            if st.button("Run Mixture Inference", type="primary"):
                x = torch.tensor(mix_intensity).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logits = mix_model(x)
                    probs = torch.sigmoid(logits).squeeze().numpy()

                pred_labels = [c for c, p in zip(class_names, probs) if p >= threshold]
                if not pred_labels:
                    pred_labels = [class_names[int(np.argmax(probs))]]

                st.write("True classes (from selected components):", ", ".join(sorted(true_labels)))
                st.write("Predicted classes:", ", ".join(pred_labels))

                prob_df = pd.DataFrame({"Class": class_names, "Probability (%)": probs * 100.0})
                fig_prob = px.bar(
                    prob_df, x="Class", y="Probability (%)", color="Class",
                    title="Per-class probabilities (sigmoid)", height=300
                )
                st.plotly_chart(fig_prob, use_container_width=True)
        except Exception as e:
            st.error(f"Mixture inference error: {e}")
