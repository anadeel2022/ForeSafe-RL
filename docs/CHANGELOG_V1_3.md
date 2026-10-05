# ForeSafe-RL V1.3

V1.3 aligns the learned risk target with the simulator's denominator-correct
reliability definition before long pilot simulations.

1. The finite-horizon predictive-risk label is now
      sum(failed scheduled transmissions) / sum(scheduled transmissions)
   over the H-slot prediction window.
   It no longer averages slot-wise failure ratios or treats each idle slot as
   an equally weighted zero-risk sample.
2. The same denominator-correct H-slot ratio is used by the online
   distribution-shift monitor.
3. Shift detection uses the predicted q50 (median) residual against the
   realized H-slot ratio. Integrated upper-tail CVaR is reserved for safety
   filtering. This separates calibration/shift detection from tail-risk control.
4. Training and evaluation record mean predicted median risk in addition to CVaR.
5. The V1.2 numerical CVaR integral, warm-up protection, physical resource mask,
   action-filter diagnostics, urgency guard, and exact PPO rollout masks are retained.
6. Structural validation includes a hand-checkable denominator-correct horizon
   target.
