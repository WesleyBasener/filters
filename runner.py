import numpy as np
from particles import GroundTruthFactory
from sensors import RBESensor, XYZSensor
from filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter
from grapher import plot_3d_lines
from utils import cart_to_rbe, rbe_to_cart
import matplotlib.pyplot as plt

def main():
    gt = GroundTruthFactory("CWNAM", 100, init=np.array([100, 0, 100, 0, 50, 0]))
    rbe_sensor = RBESensor(dim=gt.dim, num_dirs=gt.num_dirs)
    xyz_sensor = XYZSensor(dim=gt.dim, num_dirs_gt=gt.num_dirs)

    mes_xyz = xyz_sensor.measure_ground_truths(gt)
    mes_rbe = rbe_sensor.measure_ground_truths(gt)

    kf = KalmanFilter(gt, mes_xyz)
    ekf = ExtendedKalmanFilter(gt, mes_rbe)
    ukf = UnscentedKalmanFilter(gt, mes_rbe, alpha=0.2, beta=2)
    pf = ParticleFilter(gt, mes_rbe)

    kf_preds = []
    ekf_preds = []
    ukf_preds = []
    pf_preds = []

    for z in mes_xyz.measurements[1:]:
        kf.predict_and_update(z)
        kf_preds.append(kf.x)

    plt1 = plot_3d_lines(np.array(gt.ground_truth)[:, 0::(gt.num_dirs+1)], np.array(kf_preds)[:, 0::(gt.num_dirs+1)], np.array(mes_xyz.measurements), "Kalman Filter")

    for z in mes_rbe.measurements[1:]:
        ekf.predict_and_update(z)
        ekf_preds.append(ekf.x)

    plt2 = plot_3d_lines(np.array(gt.ground_truth)[:, 0::(gt.num_dirs+1)], np.array(ekf_preds)[:, 0::(gt.num_dirs+1)], np.array([rbe_to_cart(x) for x in mes_rbe.measurements]), "Extended Kalman Filter")
    
    for z in mes_rbe.measurements[1:]:
        ukf.predict_and_update(z)
        ukf_preds.append(ukf.x)

    plt3 = plot_3d_lines(np.array(gt.ground_truth)[:, 0::(gt.num_dirs+1)], np.array(ukf_preds)[:, 0::(gt.num_dirs+1)], np.array([rbe_to_cart(x) for x in mes_rbe.measurements]), "Unscented Kalman Filter")
    
    for z in mes_rbe.measurements[1:]:
        pf.predict_and_update(z)
        pf_preds.append(pf.x)

    plt4 = plot_3d_lines(np.array(gt.ground_truth)[:, 0::(gt.num_dirs+1)], np.array(pf_preds)[:, 0::(gt.num_dirs+1)], np.array([rbe_to_cart(x) for x in mes_rbe.measurements]), "Particle Filter")

    plt.show()

if __name__ == "__main__":
    main()