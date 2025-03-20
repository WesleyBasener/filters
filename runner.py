import numpy as np
from particles import GroundTruthFactory
from sensors import RBESensor, XYZSensor
from filters import KalmanFilter

def main():
    gt = GroundTruthFactory("CNWP", 100)
    rbe_sensor = RBESensor()
    xyz_sensor = XYZSensor()

    mes_xyz = xyz_sensor.measure_ground_truths(gt)
    mes_rbe = rbe_sensor.measure_ground_truths(gt)

    kf = KalmanFilter(gt, mes_xyz)

    kf_preds = []

    for z in mes_xyz.measurements:
        kf.predict_and_update(z)
        kf_preds.append(z)

    pass
    

if __name__ == "__main__":
    main()