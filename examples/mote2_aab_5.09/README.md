# 1+2 A-AB MoTe2 5.09 Degree

This directory contains one Gamma spinful model and two separate single-spin
K1 models, A and B. The K1 models share one spinful TAPW calculation but retain
different active layer sectors.

```bash
tapw run  -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
tapw symm -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
tapw run  -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml
tapw symm -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml
kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
kp symm   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
kp symm-rep -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml # optional
kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml
kp symm   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml
kp symm-rep -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml         # optional
kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml
kp symm   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml
kp symm-rep -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml         # optional
```

Projection and harmonic counts are automatic. The Gamma model uses explicit
kinetic/intralayer/interlayer orders 8/6/8; all three fits are linear.
