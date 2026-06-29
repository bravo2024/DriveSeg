from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="DriveSeg | AImotive Perception", layout="wide", page_icon="\U0001f697")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Frames",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("AImotive | Drive Segmentation | Autonomous Perception")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Frames",f"{n:,}"); c2.metric("Hazard Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f30d Segmentation Quality","\U0001f6a8 Safety"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Safe","Hazard"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Scene Hazard Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.markdown("**Per-pixel segmentation metrics:** mean_iou, per_class_iou_vehicle, per_class_iou_pedestrian, per_class_iou_road, dice_coeff, boundary_f1, edge_intensity, texture_variance, depth_variance")
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Semantic Segmentation Quality")
    st.latex(r"\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \quad \text{mIoU} = \frac{1}{C}\sum_{c=1}^C \text{IoU}_c")
    st.caption("Intersection over Union per class (vehicle, pedestrian, road, background) and mean IoU across all C classes. mIoU > 0.65 is production-grade for autonomous driving.")
    st.latex(r"\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \frac{2 \cdot |A \cap B|}{|A| + |B|}")
    st.caption("Dice coefficient (F1 at pixel level): harmonic mean of precision and recall per class. Less sensitive to class imbalance than IoU.")
    st.latex(r"\text{Boundary F1} = \frac{2 \cdot \text{Precision}_b \cdot \text{Recall}_b}{\text{Precision}_b + \text{Recall}_b}")
    st.caption("Boundary F1 measures segmentation quality at object edges. Critical for autonomous driving where 1-2 pixel boundary errors can affect distance estimation and path planning.")
    col_a,col_b=st.columns(2)
    with col_a:
        mean_iou=data["df"]["mean_iou"]
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        ax.hist(mean_iou,bins=30,color="#22d3ee",alpha=0.6,edgecolor="#1a1f2e")
        ax.axvline(mean_iou.mean(),color="#f97316",ls="--",lw=2,label=f"mIoU={mean_iou.mean():.3f}")
        ax.set_title("Mean IoU Distribution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        dice=data["df"]["dice_coeff"]
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        ax.hist(dice,bins=30,color="#a78bfa",alpha=0.6,edgecolor="#1a1f2e")
        ax.axvline(dice.mean(),color="#22c55e",ls="--",lw=2,label=f"Dice={dice.mean():.3f}")
        ax.set_title("Dice Coefficient Distribution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
with t4:
    st.subheader("Safety Metrics & BEV Perception")
    st.latex(r"\text{Risk Score} = \frac{1}{1+e^{-(\beta_0 + \beta_1 \cdot (1-\text{mIoU}) + \beta_2 \cdot \text{hazard\_prob})}}")
    st.latex(r"P_{\text{BEV}} = K^{-1} \cdot [u, v, 1]^\top \cdot z \quad (\text{inverse perspective mapping})")
    st.caption("Bird's Eye View transform: projects segmentation mask pixels (u,v) at depth z onto the ground plane using camera intrinsics K. AImotive's aiSim uses BEV representations for fusing multi-camera perception with path planning.")
    st.latex(r"K_k = P_{k \mid k-1} H_k^\top (H_k P_{k \mid k-1} H_k^\top + R_k)^{-1}")
    st.latex(r"\hat{x}_{k \mid k-1} = F_k \hat{x}_{k-1 \mid k-1},\quad \hat{x}_{k \mid k} = \hat{x}_{k \mid k-1} + K_k (z_k - H_k \hat{x}_{k \mid k-1})")
    st.caption("Kalman Filter predict-update for object tracking across frames. Critical for temporal consistency in perception — bounding boxes and segmentation masks filtered over time to reduce false positives from occlusion or noise.")
    risk_probas=b["results"]["XGBoost"]["y_proba"]
    high_risk=(risk_probas>0.7).sum(); med_risk=((risk_probas>0.4)&(risk_probas<=0.7)).sum(); low_risk=(risk_probas<=0.4).sum()
    c1,c2,c3=st.columns(3)
    c1.metric("High Risk Scenes",f"{high_risk:,}"); c2.metric("Medium Risk",f"{med_risk:,}"); c3.metric("Low Risk",f"{low_risk:,}")
    fig,ax=plt.subplots(figsize=(8,4)); _style()
    ax.hist(risk_probas,bins=50,color="#22d3ee",alpha=0.6,edgecolor="#1a1f2e")
    ax.axvline(0.4,color="#fbbf24",ls="--",lw=2,label="Medium Threshold")
    ax.axvline(0.7,color="#f43f5e",ls="--",lw=2,label="High Threshold")
    ax.set_title("Scene Risk Score Distribution (Segmentation-Aware)",color="white"); ax.legend(); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.subheader("Blur Analysis — Frame Quality")
    blur_hazard=data["df"].groupby(pd.cut(data["df"]["frame_blur"],bins=4,labels=["Sharp","Slight","Moderate","Heavy"]),observed=True)[TARGET_NAME].mean()
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(range(4),blur_hazard.values,color=["#22c55e","#fbbf24","#f97316","#f43f5e"])
    ax.set_xticks(range(4)); ax.set_xticklabels(blur_hazard.index); ax.set_title("Hazard Rate by Blur Level",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
