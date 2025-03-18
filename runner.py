import numpy as np
from particles import GroundTruthFactory
from sensors import RBESensor, XYZSensor

def main():
    gt = GroundTruthFactory("CNWP", 100)
    rbe_sensor = RBESensor()
    xyz_sensor = XYZSensor()

    mes_xyz = xyz_sensor.measure_ground_truths(gt)
    mes_rbe = rbe_sensor.measure_ground_truths(gt)

    

if __name__ == "__main__":
    main()