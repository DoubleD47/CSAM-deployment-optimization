import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd

# Paste the output text here (or read from a file)
output_text = """
Random seed set to: 123

Demands:
D(m1, t=1, ('l1', 'k1')) = 7.3
D(m1, t=1, ('l1', 'k2')) = 3.6
D(m1, t=1, ('l1', 'k3')) = 3.0
D(m1, t=1, ('l1', 'k4')) = 6.0
D(m1, t=1, ('l1', 'k5')) = 7.5
D(m1, t=1, ('l2', 'k1')) = 4.8
D(m1, t=1, ('l2', 'k2')) = 9.8
D(m1, t=1, ('l2', 'k3')) = 7.2
D(m1, t=1, ('l2', 'k4')) = 5.3
D(m1, t=1, ('l2', 'k5')) = 4.5
D(m1, t=2, ('l1', 'k1')) = 4.1
D(m1, t=2, ('l1', 'k2')) = 7.6
D(m1, t=2, ('l1', 'k3')) = 4.9
D(m1, t=2, ('l1', 'k4')) = 1.5
D(m1, t=2, ('l1', 'k5')) = 4.6
D(m1, t=2, ('l2', 'k1')) = 7.6
D(m1, t=2, ('l2', 'k2')) = 2.6
D(m1, t=2, ('l2', 'k3')) = 2.6
D(m1, t=2, ('l2', 'k4')) = 5.8
D(m1, t=2, ('l2', 'k5')) = 5.8
D(m2, t=1, ('l1', 'k1')) = 6.7
D(m2, t=1, ('l1', 'k2')) = 8.6
D(m2, t=1, ('l1', 'k3')) = 7.5
D(m2, t=1, ('l1', 'k4')) = 6.5
D(m2, t=1, ('l1', 'k5')) = 7.5
D(m2, t=1, ('l2', 'k1')) = 3.9
D(m2, t=1, ('l2', 'k2')) = 4.3
D(m2, t=1, ('l2', 'k3')) = 3.1
D(m2, t=1, ('l2', 'k4')) = 3.6
D(m2, t=1, ('l2', 'k5')) = 6.7
D(m2, t=2, ('l1', 'k1')) = 1.8
D(m2, t=2, ('l1', 'k2')) = 4.9
D(m2, t=2, ('l1', 'k3')) = 4.9
D(m2, t=2, ('l1', 'k4')) = 5.4
D(m2, t=2, ('l1', 'k5')) = 4.8
D(m2, t=2, ('l2', 'k1')) = 3.8
D(m2, t=2, ('l2', 'k2')) = 4.8
D(m2, t=2, ('l2', 'k3')) = 9.0
D(m2, t=2, ('l2', 'k4')) = 9.5
D(m2, t=2, ('l2', 'k5')) = 5.5
D(m3, t=1, ('l1', 'k1')) = 6.6
D(m3, t=1, ('l1', 'k2')) = 2.0
D(m3, t=1, ('l1', 'k3')) = 3.9
D(m3, t=1, ('l1', 'k4')) = 4.7
D(m3, t=1, ('l1', 'k5')) = 8.8
D(m3, t=1, ('l2', 'k1')) = 3.3
D(m3, t=1, ('l2', 'k2')) = 5.3
D(m3, t=1, ('l2', 'k3')) = 9.9
D(m3, t=1, ('l2', 'k4')) = 5.7
D(m3, t=1, ('l2', 'k5')) = 6.5
D(m3, t=2, ('l1', 'k1')) = 2.1
D(m3, t=2, ('l1', 'k2')) = 8.4
D(m3, t=2, ('l1', 'k3')) = 6.4
D(m3, t=2, ('l1', 'k4')) = 5.9
D(m3, t=2, ('l1', 'k5')) = 4.1
D(m3, t=2, ('l2', 'k1')) = 3.7
D(m3, t=2, ('l2', 'k2')) = 4.8
D(m3, t=2, ('l2', 'k3')) = 7.1
D(m3, t=2, ('l2', 'k4')) = 8.9
D(m3, t=2, ('l2', 'k5')) = 5.6
D(m4, t=1, ('l1', 'k1')) = 7.0
D(m4, t=1, ('l1', 'k2')) = 6.3
D(m4, t=1, ('l1', 'k3')) = 6.6
D(m4, t=1, ('l1', 'k4')) = 7.1
D(m4, t=1, ('l1', 'k5')) = 8.6
D(m4, t=1, ('l2', 'k1')) = 1.7
D(m4, t=1, ('l2', 'k2')) = 7.9
D(m4, t=1, ('l2', 'k3')) = 3.2
D(m4, t=1, ('l2', 'k4')) = 2.7
D(m4, t=1, ('l2', 'k5')) = 6.2
D(m4, t=2, ('l1', 'k1')) = 1.9
D(m4, t=2, ('l1', 'k2')) = 9.0
D(m4, t=2, ('l1', 'k3')) = 6.6
D(m4, t=2, ('l1', 'k4')) = 7.5
D(m4, t=2, ('l1', 'k5')) = 1.1
D(m4, t=2, ('l2', 'k1')) = 6.3
D(m4, t=2, ('l2', 'k2')) = 6.0
D(m4, t=2, ('l2', 'k3')) = 2.4
D(m4, t=2, ('l2', 'k4')) = 2.4
D(m4, t=2, ('l2', 'k5')) = 7.3
D(m5, t=1, ('l1', 'k1')) = 3.9
D(m5, t=1, ('l1', 'k2')) = 7.2
D(m5, t=1, ('l1', 'k3')) = 6.0
D(m5, t=1, ('l1', 'k4')) = 4.5
D(m5, t=1, ('l1', 'k5')) = 9.3
D(m5, t=1, ('l2', 'k1')) = 8.6
D(m5, t=1, ('l2', 'k2')) = 4.2
D(m5, t=1, ('l2', 'k3')) = 1.4
D(m5, t=1, ('l2', 'k4')) = 3.7
D(m5, t=1, ('l2', 'k5')) = 4.6
D(m5, t=2, ('l1', 'k1')) = 7.3
D(m5, t=2, ('l1', 'k2')) = 10.0
D(m5, t=2, ('l1', 'k3')) = 4.2
D(m5, t=2, ('l1', 'k4')) = 7.9
D(m5, t=2, ('l1', 'k5')) = 6.3
D(m5, t=2, ('l2', 'k1')) = 7.2
D(m5, t=2, ('l2', 'k2')) = 2.4
D(m5, t=2, ('l2', 'k3')) = 4.6
D(m5, t=2, ('l2', 'k4')) = 3.2
D(m5, t=2, ('l2', 'k5')) = 4.1
D(m6, t=1, ('l1', 'k1')) = 5.6
D(m6, t=1, ('l1', 'k2')) = 7.0
D(m6, t=1, ('l1', 'k3')) = 2.0
D(m6, t=1, ('l1', 'k4')) = 2.2
D(m6, t=1, ('l1', 'k5')) = 3.9
D(m6, t=1, ('l2', 'k1')) = 7.0
D(m6, t=1, ('l2', 'k2')) = 8.6
D(m6, t=1, ('l2', 'k3')) = 6.0
D(m6, t=1, ('l2', 'k4')) = 8.7
D(m6, t=1, ('l2', 'k5')) = 4.5
D(m6, t=2, ('l1', 'k1')) = 3.9
D(m6, t=2, ('l1', 'k2')) = 4.2
D(m6, t=2, ('l1', 'k3')) = 2.5
D(m6, t=2, ('l1', 'k4')) = 8.5
D(m6, t=2, ('l1', 'k5')) = 4.0
D(m6, t=2, ('l2', 'k1')) = 6.0
D(m6, t=2, ('l2', 'k2')) = 6.2
D(m6, t=2, ('l2', 'k3')) = 5.7
D(m6, t=2, ('l2', 'k4')) = 1.0
D(m6, t=2, ('l2', 'k5')) = 9.9
D(m7, t=1, ('l1', 'k1')) = 9.1
D(m7, t=1, ('l1', 'k2')) = 2.9
D(m7, t=1, ('l1', 'k3')) = 3.6
D(m7, t=1, ('l1', 'k4')) = 5.7
D(m7, t=1, ('l1', 'k5')) = 9.1
D(m7, t=1, ('l2', 'k1')) = 9.9
D(m7, t=1, ('l2', 'k2')) = 3.3
D(m7, t=1, ('l2', 'k3')) = 6.1
D(m7, t=1, ('l2', 'k4')) = 8.3
D(m7, t=1, ('l2', 'k5')) = 4.5
D(m7, t=2, ('l1', 'k1')) = 7.6
D(m7, t=2, ('l1', 'k2')) = 2.4
D(m7, t=2, ('l1', 'k3')) = 6.4
D(m7, t=2, ('l1', 'k4')) = 8.8
D(m7, t=2, ('l1', 'k5')) = 9.9
D(m7, t=2, ('l2', 'k1')) = 1.7
D(m7, t=2, ('l2', 'k2')) = 4.9
D(m7, t=2, ('l2', 'k3')) = 2.8
D(m7, t=2, ('l2', 'k4')) = 5.1
D(m7, t=2, ('l2', 'k5')) = 5.9
D(m8, t=1, ('l1', 'k1')) = 1.8
D(m8, t=1, ('l1', 'k2')) = 3.7
D(m8, t=1, ('l1', 'k3')) = 9.3
D(m8, t=1, ('l1', 'k4')) = 6.1
D(m8, t=1, ('l1', 'k5')) = 5.1
D(m8, t=1, ('l2', 'k1')) = 7.8
D(m8, t=1, ('l2', 'k2')) = 7.7
D(m8, t=1, ('l2', 'k3')) = 1.4
D(m8, t=1, ('l2', 'k4')) = 7.4
D(m8, t=1, ('l2', 'k5')) = 8.6
D(m8, t=2, ('l1', 'k1')) = 2.5
D(m8, t=2, ('l1', 'k2')) = 8.0
D(m8, t=2, ('l1', 'k3')) = 3.6
D(m8, t=2, ('l1', 'k4')) = 3.8
D(m8, t=2, ('l1', 'k5')) = 7.0
D(m8, t=2, ('l2', 'k1')) = 2.0
D(m8, t=2, ('l2', 'k2')) = 7.0
D(m8, t=2, ('l2', 'k3')) = 9.0
D(m8, t=2, ('l2', 'k4')) = 7.3
D(m8, t=2, ('l2', 'k5')) = 5.0
D(m9, t=1, ('l1', 'k1')) = 4.9
D(m9, t=1, ('l1', 'k2')) = 7.9
D(m9, t=1, ('l1', 'k3')) = 6.1
D(m9, t=1, ('l1', 'k4')) = 1.8
D(m9, t=1, ('l1', 'k5')) = 6.2
D(m9, t=1, ('l2', 'k1')) = 8.3
D(m9, t=1, ('l2', 'k2')) = 4.0
D(m9, t=1, ('l2', 'k3')) = 9.3
D(m9, t=1, ('l2', 'k4')) = 7.8
D(m9, t=1, ('l2', 'k5')) = 6.2
D(m9, t=2, ('l1', 'k1')) = 7.8
D(m9, t=2, ('l1', 'k2')) = 1.7
D(m9, t=2, ('l1', 'k3')) = 8.7
D(m9, t=2, ('l1', 'k4')) = 8.4
D(m9, t=2, ('l1', 'k5')) = 9.2
D(m9, t=2, ('l2', 'k1')) = 2.2
D(m9, t=2, ('l2', 'k2')) = 1.7
D(m9, t=2, ('l2', 'k3')) = 2.2
D(m9, t=2, ('l2', 'k4')) = 4.6
D(m9, t=2, ('l2', 'k5')) = 4.8
D(m10, t=1, ('l1', 'k1')) = 6.1
D(m10, t=1, ('l1', 'k2')) = 2.1
D(m10, t=1, ('l1', 'k3')) = 2.8
D(m10, t=1, ('l1', 'k4')) = 8.3
D(m10, t=1, ('l1', 'k5')) = 5.2
D(m10, t=1, ('l2', 'k1')) = 8.3
D(m10, t=1, ('l2', 'k2')) = 1.1
D(m10, t=1, ('l2', 'k3')) = 6.0
D(m10, t=1, ('l2', 'k4')) = 9.4
D(m10, t=1, ('l2', 'k5')) = 6.2
D(m10, t=2, ('l1', 'k1')) = 2.9
D(m10, t=2, ('l1', 'k2')) = 7.5
D(m10, t=2, ('l1', 'k3')) = 4.4
D(m10, t=2, ('l1', 'k4')) = 7.0
D(m10, t=2, ('l1', 'k5')) = 1.3
D(m10, t=2, ('l2', 'k1')) = 6.7
D(m10, t=2, ('l2', 'k2')) = 1.3
D(m10, t=2, ('l2', 'k3')) = 7.7
D(m10, t=2, ('l2', 'k4')) = 5.3
D(m10, t=2, ('l2', 'k5')) = 2.1
Solving model...
Status: Optimal
Objective: 13343.725417556823

Facility openings (CSAM l1):
y[m1, 'l1'] = 1.0
y[m2, 'l1'] = 1.0
y[m3, 'l1'] = 1.0
y[m4, 'l1'] = 1.0
y[m5, 'l1'] = 1.0
y[m6, 'l1'] = 1.0
y[m7, 'l1'] = 1.0
y[m8, 'l1'] = 1.0
y[m9, 'l1'] = 1.0
y[m10, 'l1'] = 1.0

Positive flows on CSAM (l1) repair paths (only flexible l1 commodities):
Arc (m1_q_l1 -> m1_r_l1), t=1, commodity=('l1', 'k1'): flow=7.3
Arc (m1_q_l1 -> m1_r_l1), t=1, commodity=('l1', 'k2'): flow=3.6
Arc (m1_q_l1 -> m1_r_l1), t=1, commodity=('l1', 'k3'): flow=3.0
Arc (m1_q_l1 -> m1_r_l1), t=1, commodity=('l1', 'k4'): flow=6.0
Arc (m1_q_l1 -> m1_r_l1), t=1, commodity=('l1', 'k5'): flow=7.5
Arc (m1_q_l1 -> m1_r_l1), t=2, commodity=('l1', 'k1'): flow=4.1
Arc (m1_q_l1 -> m1_r_l1), t=2, commodity=('l1', 'k2'): flow=7.6
Arc (m1_q_l1 -> m1_r_l1), t=2, commodity=('l1', 'k3'): flow=4.9
Arc (m1_q_l1 -> m1_r_l1), t=2, commodity=('l1', 'k4'): flow=1.5
Arc (m1_q_l1 -> m1_r_l1), t=2, commodity=('l1', 'k5'): flow=4.6
Arc (m2_q_l1 -> m2_r_l1), t=1, commodity=('l1', 'k1'): flow=6.7
Arc (m2_q_l1 -> m2_r_l1), t=1, commodity=('l1', 'k2'): flow=8.6
Arc (m2_q_l1 -> m2_r_l1), t=1, commodity=('l1', 'k3'): flow=7.5
Arc (m2_q_l1 -> m2_r_l1), t=1, commodity=('l1', 'k4'): flow=6.5
Arc (m2_q_l1 -> m2_r_l1), t=1, commodity=('l1', 'k5'): flow=7.5
Arc (m2_q_l1 -> m2_r_l1), t=2, commodity=('l1', 'k1'): flow=1.8
Arc (m2_q_l1 -> m2_r_l1), t=2, commodity=('l1', 'k2'): flow=4.9
Arc (m2_q_l1 -> m2_r_l1), t=2, commodity=('l1', 'k3'): flow=4.9
Arc (m2_q_l1 -> m2_r_l1), t=2, commodity=('l1', 'k4'): flow=5.4
Arc (m2_q_l1 -> m2_r_l1), t=2, commodity=('l1', 'k5'): flow=4.8
Arc (m3_q_l1 -> m3_r_l1), t=1, commodity=('l1', 'k1'): flow=6.6
Arc (m3_q_l1 -> m3_r_l1), t=1, commodity=('l1', 'k2'): flow=2.0
Arc (m3_q_l1 -> m3_r_l1), t=1, commodity=('l1', 'k3'): flow=3.9
Arc (m3_q_l1 -> m3_r_l1), t=1, commodity=('l1', 'k4'): flow=4.7
Arc (m3_q_l1 -> m3_r_l1), t=1, commodity=('l1', 'k5'): flow=8.8
Arc (m3_q_l1 -> m3_r_l1), t=2, commodity=('l1', 'k1'): flow=2.1
Arc (m3_q_l1 -> m3_r_l1), t=2, commodity=('l1', 'k2'): flow=8.4
Arc (m3_q_l1 -> m3_r_l1), t=2, commodity=('l1', 'k3'): flow=6.4
Arc (m3_q_l1 -> m3_r_l1), t=2, commodity=('l1', 'k4'): flow=5.9
Arc (m3_q_l1 -> m3_r_l1), t=2, commodity=('l1', 'k5'): flow=4.1
Arc (m4_q_l1 -> m4_r_l1), t=1, commodity=('l1', 'k1'): flow=7.0
Arc (m4_q_l1 -> m4_r_l1), t=1, commodity=('l1', 'k2'): flow=6.3
Arc (m4_q_l1 -> m4_r_l1), t=1, commodity=('l1', 'k3'): flow=6.6
Arc (m4_q_l1 -> m4_r_l1), t=1, commodity=('l1', 'k4'): flow=7.1
Arc (m4_q_l1 -> m4_r_l1), t=1, commodity=('l1', 'k5'): flow=8.6
Arc (m4_q_l1 -> m4_r_l1), t=2, commodity=('l1', 'k1'): flow=1.9
Arc (m4_q_l1 -> m4_r_l1), t=2, commodity=('l1', 'k2'): flow=9.0
Arc (m4_q_l1 -> m4_r_l1), t=2, commodity=('l1', 'k3'): flow=6.6
Arc (m4_q_l1 -> m4_r_l1), t=2, commodity=('l1', 'k4'): flow=7.5
Arc (m4_q_l1 -> m4_r_l1), t=2, commodity=('l1', 'k5'): flow=1.1
Arc (m5_q_l1 -> m5_r_l1), t=1, commodity=('l1', 'k1'): flow=3.9
Arc (m5_q_l1 -> m5_r_l1), t=1, commodity=('l1', 'k2'): flow=7.2
Arc (m5_q_l1 -> m5_r_l1), t=1, commodity=('l1', 'k3'): flow=6.0
Arc (m5_q_l1 -> m5_r_l1), t=1, commodity=('l1', 'k4'): flow=4.5
Arc (m5_q_l1 -> m5_r_l1), t=1, commodity=('l1', 'k5'): flow=9.3
Arc (m5_q_l1 -> m5_r_l1), t=2, commodity=('l1', 'k1'): flow=7.3
Arc (m5_q_l1 -> m5_r_l1), t=2, commodity=('l1', 'k2'): flow=10.0
Arc (m5_q_l1 -> m5_r_l1), t=2, commodity=('l1', 'k3'): flow=4.2
Arc (m5_q_l1 -> m5_r_l1), t=2, commodity=('l1', 'k4'): flow=7.9
Arc (m5_q_l1 -> m5_r_l1), t=2, commodity=('l1', 'k5'): flow=6.3
Arc (m6_q_l1 -> m6_r_l1), t=1, commodity=('l1', 'k1'): flow=5.6
Arc (m6_q_l1 -> m6_r_l1), t=1, commodity=('l1', 'k2'): flow=7.0
Arc (m6_q_l1 -> m6_r_l1), t=1, commodity=('l1', 'k3'): flow=2.0
Arc (m6_q_l1 -> m6_r_l1), t=1, commodity=('l1', 'k4'): flow=2.2
Arc (m6_q_l1 -> m6_r_l1), t=1, commodity=('l1', 'k5'): flow=3.9
Arc (m6_q_l1 -> m6_r_l1), t=2, commodity=('l1', 'k1'): flow=3.9
Arc (m6_q_l1 -> m6_r_l1), t=2, commodity=('l1', 'k2'): flow=4.2
Arc (m6_q_l1 -> m6_r_l1), t=2, commodity=('l1', 'k3'): flow=2.5
Arc (m6_q_l1 -> m6_r_l1), t=2, commodity=('l1', 'k4'): flow=8.5
Arc (m6_q_l1 -> m6_r_l1), t=2, commodity=('l1', 'k5'): flow=4.0
Arc (m7_q_l1 -> m7_r_l1), t=1, commodity=('l1', 'k1'): flow=9.1
Arc (m7_q_l1 -> m7_r_l1), t=1, commodity=('l1', 'k2'): flow=2.9
Arc (m7_q_l1 -> m7_r_l1), t=1, commodity=('l1', 'k3'): flow=3.6
Arc (m7_q_l1 -> m7_r_l1), t=1, commodity=('l1', 'k4'): flow=5.7
Arc (m7_q_l1 -> m7_r_l1), t=1, commodity=('l1', 'k5'): flow=9.1
Arc (m7_q_l1 -> m7_r_l1), t=2, commodity=('l1', 'k1'): flow=7.6
Arc (m7_q_l1 -> m7_r_l1), t=2, commodity=('l1', 'k2'): flow=2.4
Arc (m7_q_l1 -> m7_r_l1), t=2, commodity=('l1', 'k3'): flow=6.4
Arc (m7_q_l1 -> m7_r_l1), t=2, commodity=('l1', 'k4'): flow=8.8
Arc (m7_q_l1 -> m7_r_l1), t=2, commodity=('l1', 'k5'): flow=9.9
Arc (m8_q_l1 -> m8_r_l1), t=1, commodity=('l1', 'k1'): flow=1.8
Arc (m8_q_l1 -> m8_r_l1), t=1, commodity=('l1', 'k2'): flow=3.7
Arc (m8_q_l1 -> m8_r_l1), t=1, commodity=('l1', 'k3'): flow=9.3
Arc (m8_q_l1 -> m8_r_l1), t=1, commodity=('l1', 'k4'): flow=6.1
Arc (m8_q_l1 -> m8_r_l1), t=1, commodity=('l1', 'k5'): flow=5.1
Arc (m8_q_l1 -> m8_r_l1), t=2, commodity=('l1', 'k1'): flow=2.5
Arc (m8_q_l1 -> m8_r_l1), t=2, commodity=('l1', 'k2'): flow=8.0
Arc (m8_q_l1 -> m8_r_l1), t=2, commodity=('l1', 'k3'): flow=3.6
Arc (m8_q_l1 -> m8_r_l1), t=2, commodity=('l1', 'k4'): flow=3.8
Arc (m8_q_l1 -> m8_r_l1), t=2, commodity=('l1', 'k5'): flow=7.0
Arc (m9_q_l1 -> m9_r_l1), t=1, commodity=('l1', 'k1'): flow=4.9
Arc (m9_q_l1 -> m9_r_l1), t=1, commodity=('l1', 'k2'): flow=7.9
Arc (m9_q_l1 -> m9_r_l1), t=1, commodity=('l1', 'k3'): flow=6.1
Arc (m9_q_l1 -> m9_r_l1), t=1, commodity=('l1', 'k4'): flow=1.8
Arc (m9_q_l1 -> m9_r_l1), t=1, commodity=('l1', 'k5'): flow=6.2
Arc (m9_q_l1 -> m9_r_l1), t=2, commodity=('l1', 'k1'): flow=7.8
Arc (m9_q_l1 -> m9_r_l1), t=2, commodity=('l1', 'k2'): flow=1.7
Arc (m9_q_l1 -> m9_r_l1), t=2, commodity=('l1', 'k3'): flow=8.7
Arc (m9_q_l1 -> m9_r_l1), t=2, commodity=('l1', 'k4'): flow=8.4
Arc (m9_q_l1 -> m9_r_l1), t=2, commodity=('l1', 'k5'): flow=9.2
Arc (m10_q_l1 -> m10_r_l1), t=1, commodity=('l1', 'k1'): flow=6.1
Arc (m10_q_l1 -> m10_r_l1), t=1, commodity=('l1', 'k2'): flow=2.1
Arc (m10_q_l1 -> m10_r_l1), t=1, commodity=('l1', 'k3'): flow=2.8
Arc (m10_q_l1 -> m10_r_l1), t=1, commodity=('l1', 'k4'): flow=8.3
Arc (m10_q_l1 -> m10_r_l1), t=1, commodity=('l1', 'k5'): flow=5.2
Arc (m10_q_l1 -> m10_r_l1), t=2, commodity=('l1', 'k1'): flow=2.9
Arc (m10_q_l1 -> m10_r_l1), t=2, commodity=('l1', 'k2'): flow=7.5
Arc (m10_q_l1 -> m10_r_l1), t=2, commodity=('l1', 'k3'): flow=4.4
Arc (m10_q_l1 -> m10_r_l1), t=2, commodity=('l1', 'k4'): flow=7.0
Arc (m10_q_l1 -> m10_r_l1), t=2, commodity=('l1', 'k5'): flow=1.3

Positive flows on traditional (l2) repair paths:

For k=k1 at m1:
Arc (m1_q_l2 -> m1_r_l2), t=1, commodity=('l2', 'k1'): flow=63.5 (jumping if 'l1')
Arc (m1_q_l2 -> m1_r_l2), t=2, commodity=('l2', 'k1'): flow=47.3 (jumping if 'l1')

For k=k2 at m2:
Arc (m2_q_l2 -> m2_r_l2), t=1, commodity=('l2', 'k2'): flow=56.2 (jumping if 'l1')
Arc (m2_q_l2 -> m2_r_l2), t=2, commodity=('l2', 'k2'): flow=41.7 (jumping if 'l1')

For k=k3 at m3:
Arc (m3_q_l2 -> m3_r_l2), t=1, commodity=('l2', 'k3'): flow=53.5 (jumping if 'l1')
Arc (m3_q_l2 -> m3_r_l2), t=2, commodity=('l2', 'k3'): flow=53.2 (jumping if 'l1')

For k=k4 at m4:
Arc (m4_q_l2 -> m4_r_l2), t=1, commodity=('l2', 'k4'): flow=62.6 (jumping if 'l1')
Arc (m4_q_l2 -> m4_r_l2), t=2, commodity=('l2', 'k4'): flow=52.9 (jumping if 'l1')

For k=k5 at m5:
Arc (m5_q_l2 -> m5_r_l2), t=1, commodity=('l2', 'k5'): flow=58.4 (jumping if 'l1')
Arc (m5_q_l2 -> m5_r_l2), t=2, commodity=('l2', 'k5'): flow=56.0 (jumping if 'l1')

Positive inter-facility travel flows (in-to-in):
Arc (m1_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=9.8
Arc (m1_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=2.6
Arc (m1_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=7.2
Arc (m1_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=2.6
Arc (m1_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=5.3
Arc (m1_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=5.8
Arc (m1_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=4.5
Arc (m1_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=5.8
Arc (m2_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=3.9
Arc (m2_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=3.8
Arc (m2_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=3.1
Arc (m2_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=9.0
Arc (m2_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=3.6
Arc (m2_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=9.5
Arc (m2_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=6.7
Arc (m2_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=5.5
Arc (m3_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=3.3
Arc (m3_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=3.7
Arc (m3_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=5.3
Arc (m3_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=4.8
Arc (m3_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=5.7
Arc (m3_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=8.9
Arc (m3_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=6.5
Arc (m3_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=5.6
Arc (m4_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=1.7
Arc (m4_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=6.3
Arc (m4_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=7.9
Arc (m4_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=6.0
Arc (m4_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=3.2
Arc (m4_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=2.4
Arc (m4_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=6.2
Arc (m4_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=7.3
Arc (m5_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=8.6
Arc (m5_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=7.2
Arc (m5_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=4.2
Arc (m5_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=2.4
Arc (m5_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=1.4
Arc (m5_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=4.6
Arc (m5_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=3.7
Arc (m5_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=3.2
Arc (m6_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=7.0
Arc (m6_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=6.0
Arc (m6_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=8.6
Arc (m6_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=6.2
Arc (m6_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=6.0
Arc (m6_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=5.7
Arc (m6_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=8.7
Arc (m6_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=1.0
Arc (m6_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=4.5
Arc (m6_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=9.9
Arc (m7_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=9.9
Arc (m7_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=1.7
Arc (m7_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=3.3
Arc (m7_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=4.9
Arc (m7_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=6.1
Arc (m7_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=2.8
Arc (m7_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=8.3
Arc (m7_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=5.1
Arc (m7_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=4.5
Arc (m7_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=5.9
Arc (m8_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=7.8
Arc (m8_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=2.0
Arc (m8_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=7.7
Arc (m8_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=7.0
Arc (m8_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=1.4
Arc (m8_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=9.0
Arc (m8_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=7.4
Arc (m8_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=7.3
Arc (m8_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=8.6
Arc (m8_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=5.0
Arc (m9_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=8.3
Arc (m9_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=2.2
Arc (m9_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=4.0
Arc (m9_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=1.7
Arc (m9_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=9.3
Arc (m9_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=2.2
Arc (m9_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=7.8
Arc (m9_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=4.6
Arc (m9_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=6.2
Arc (m9_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=4.8
Arc (m10_in -> m1_in), t=1, commodity=('l2', 'k1'): flow=8.3
Arc (m10_in -> m1_in), t=2, commodity=('l2', 'k1'): flow=6.7
Arc (m10_in -> m2_in), t=1, commodity=('l2', 'k2'): flow=1.1
Arc (m10_in -> m2_in), t=2, commodity=('l2', 'k2'): flow=1.3
Arc (m10_in -> m3_in), t=1, commodity=('l2', 'k3'): flow=6.0
Arc (m10_in -> m3_in), t=2, commodity=('l2', 'k3'): flow=7.7
Arc (m10_in -> m4_in), t=1, commodity=('l2', 'k4'): flow=9.4
Arc (m10_in -> m4_in), t=2, commodity=('l2', 'k4'): flow=5.3
Arc (m10_in -> m5_in), t=1, commodity=('l2', 'k5'): flow=6.2
Arc (m10_in -> m5_in), t=2, commodity=('l2', 'k5'): flow=2.1

Positive flows on dummy arcs (unmet demand in t=2):

Total demand: 1105.9
Total inflow to ss: 1105.9

"""

# Parse deployments, repair flows, travel flows, and dummy flows
# This is a simplified parser; adjust for full output
deployments = {'m1': 1.0, 'm2': 1.0, 'm3': 1.0}  # From output
# Example flows (extract more from output lines starting with 'Arc')
edges = [
    ('m1_in', 'm1_q_l1', {'flow': 21.8, 't':1, 'c':('l1','k1'), 'label':'CSAM Repair'}),
    ('m1_in', 'm2_in', {'flow':17.9, 't':1, 'c':('l2','k2'), 'label':'Travel'}),
    # Add all positive flows from output sections
    ('m1_q_l2', 'dummy', {'flow':64.5, 't':2, 'c':('l2','k1'), 'label':'Unmet'})
]

# Build graph
G = nx.DiGraph()
for src, dst, data in edges:
    G.add_edge(src, dst, weight=data['flow'], label=data['label'])

# Highlight opened CSAM nodes
node_colors = ['green' if 'l1' in node and deployments.get(node.split('_')[0], 0) == 1 else 'lightblue' for node in G.nodes()]

# Draw (filter by t=1 for simplicity; repeat for t=2)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('CSAM Deployment Network Flow (t=1 Example)')
plt.savefig('csam_network.png')  # View this file
plt.show()