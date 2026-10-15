#!/bin/sh
python test_avalanche_lib.py --benchmark "soft_robot" --dataset_root "${RTCL_SOFT_ROBOT_ROOT:-$HOME/.avalanche/data/soft_robot_data_raw}" --scenario_soft_robot "il" --algorithm "replay"
python test_avalanche_lib.py --benchmark "soft_robot" --dataset_root "${RTCL_SOFT_ROBOT_ROOT:-$HOME/.avalanche/data/soft_robot_data_raw}" --scenario_soft_robot "ic" --algorithm "replay"