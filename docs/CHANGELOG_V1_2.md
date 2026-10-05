# ForeSafe-RL V1.2

V1.2 is the pre-pilot methodological cleanup.

1. Replaces the equal-weight "tail quantile mean" with a numerical approximation of
   CVaR_alpha = (1/(1-alpha)) integral_alpha^1 Q(u) du.
2. Adds q=0.99 to resolve more of the extreme upper tail before the endpoint approximation.
3. Predictive-risk reward penalties are disabled during the configured warm-up period; the
   untrained predictor can no longer distort early PPO learning.
4. Separates action-filter diagnostics into:
   native resource-feasible actions,
   CVaR-admissible actions,
   final safe actions after urgency guarding,
   actions removed by CVaR,
   and actions removed by urgency.
5. Retains the exact published MobiSafe resource mask and horizon-aligned shift monitor from V1.1.
6. Structural validation now checks the CVaR integral on a constant quantile function.
