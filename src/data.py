from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["mean_iou","per_class_iou_vehicle","per_class_iou_pedestrian","per_class_iou_road","dice_coeff","boundary_f1","edge_intensity","texture_variance","road_area_pct","horizon_position","weather_score","lighting_score","frame_blur","depth_variance"]
CATEGORICAL_FEATURES = ["weather_score"]
NUMERICAL_FEATURES = ["mean_iou","per_class_iou_vehicle","per_class_iou_pedestrian","per_class_iou_road","dice_coeff","boundary_f1","edge_intensity","texture_variance","road_area_pct","horizon_position","lighting_score","frame_blur","depth_variance"]
TARGET_NAME = "hazard_detected"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "mean_iou": rng.beta(6,3,size=n).round(3),
        "per_class_iou_vehicle": rng.beta(5,4,size=n).round(3),
        "per_class_iou_pedestrian": rng.beta(4,5,size=n).round(3),
        "per_class_iou_road": rng.beta(8,2,size=n).round(3),
        "dice_coeff": rng.beta(7,2,size=n).round(3),
        "boundary_f1": rng.beta(5,3,size=n).round(3),
        "edge_intensity": rng.uniform(0,100,size=n).round(1),
        "texture_variance": rng.uniform(0,100,size=n).round(1),
        "road_area_pct": rng.beta(6,2,size=n).round(3),
        "horizon_position": rng.uniform(0.3,0.7,size=n).round(3),
        "weather_score": rng.choice([1,2,3,4,5],size=n,p=[0.30,0.30,0.20,0.12,0.08]),
        "lighting_score": rng.uniform(0,100,size=n).round(1),
        "frame_blur": rng.beta(2,8,size=n).round(3),
        "depth_variance": rng.uniform(0,100,size=n).round(1),
    })
    miou=df["mean_iou"]; iou_veh=df["per_class_iou_vehicle"]; iou_ped=df["per_class_iou_pedestrian"]
    iou_road=df["per_class_iou_road"]; dice=df["dice_coeff"]; bf1=df["boundary_f1"]
    road=df["road_area_pct"]; edge=df["edge_intensity"]/100; tex=df["texture_variance"]/100
    weather=df["weather_score"]/5; light=df["lighting_score"]/100; blur=df["frame_blur"]
    depth=df["depth_variance"]/100
    log_odds = -2.5 - 0.5*miou - 0.3*iou_veh + 0.6*(1-iou_ped) - 0.2*iou_road - 0.4*dice - 0.3*bf1 + 0.3*edge + 0.2*tex - 0.2*road + 0.1*weather - 0.2*light + 0.4*blur + 0.3*depth + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,85)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(hazard_detected=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
