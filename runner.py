import numpy as np
from particles import GroundTruthFactory
from sensors import RBESensor, XYZSensor
from filters import KalmanFilter, UnscentedKalmanFilter
from grapher import plot_3d_lines
from utils import cart_to_rbe, rbe_to_cart

def main():
    gt = GroundTruthFactory("CNWP", 100, init=np.array([100, 0, 100, 0, 50, 0]))
    rbe_sensor = RBESensor()
    xyz_sensor = XYZSensor()

    mes_xyz = xyz_sensor.measure_ground_truths(gt)
    mes_rbe = rbe_sensor.measure_ground_truths(gt)

    kf = KalmanFilter(gt, mes_xyz)
    ukf = UnscentedKalmanFilter(gt, mes_rbe, alpha=0.2, beta=2)

    kf_preds = []
    ukf_preds = []

    

    for z in mes_xyz.measurements[1:]:
        kf.predict_and_update(z)
        kf_preds.append(kf.x)

    plot_3d_lines(np.array(gt.ground_truth)[:, 0::2], np.array(kf_preds)[:, 0::2], np.array(mes_xyz.measurements))

    for z in mes_rbe.measurements[1:]:
        ukf.predict_and_update(z)
        ukf_preds.append(ukf.x)

    plot_3d_lines(np.array(gt.ground_truth)[:, 0::2], np.array(ukf_preds)[:, 0::2], np.array([rbe_to_cart(x) for x in mes_rbe.measurements]))
    

if __name__ == "__main__":
    main()