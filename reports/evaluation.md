# Offline evaluation

Held-out synthetic listings: **2,000** (118 violations, 5.9% base rate).

| system              |   PR-AUC |   ROC-AUC |   Recall @1% review |   Recall @10% review |   Block precision |   False-block rate |   Total recall |
|:--------------------|---------:|----------:|--------------------:|---------------------:|------------------:|-------------------:|---------------:|
| Rules only          |    0.382 |     0.834 |               0.119 |                0.508 |             1     |              0     |          0.661 |
| ML only             |    0.663 |     0.916 |               0.169 |                0.746 |             1     |              0     |          0.703 |
| Hybrid (TrustGuard) |    0.669 |     0.918 |               0.169 |                0.746 |             0.943 |              0.001 |          0.797 |

## Recall by violation type

| violation_type   |   recall |
|:-----------------|---------:|
| deceptive_ads    |    0.607 |
| financial_scam   |    0.794 |
| malware          |    0.885 |
| impersonation    |    0.9   |
