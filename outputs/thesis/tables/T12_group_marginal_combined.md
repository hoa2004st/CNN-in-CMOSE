**T12. Per-group marginal effect of encoder choice on mean QWK, in-domain (CMOSE) and pooled over the unseen-target cells.**

| Group | Regime | TCN | Transformer | LSTM | Spread | Prefers |
| --- | --- | --- | --- | --- | --- | --- |
| Gaze | In-domain | 0.539 | 0.537 | 0.534 | 0.006 | TCN |
| Gaze | Unseen targets | 0.093 | 0.095 | 0.094 | 0.002 | Transformer |
| Eye landmarks | In-domain | 0.538 | 0.537 | 0.535 | 0.003 | TCN |
| Eye landmarks | Unseen targets | 0.096 | 0.092 | 0.094 | 0.004 | TCN |
| Face landmarks | In-domain | 0.537 | 0.534 | 0.539 | 0.006 | LSTM |
| Face landmarks | Unseen targets | 0.092 | 0.096 | 0.093 | 0.004 | Transformer |
| Head pose | In-domain | 0.545 | 0.523 | 0.541 | 0.022 | TCN |
| Head pose | Unseen targets | 0.1 | 0.091 | 0.09 | 0.01 | TCN |
| Action units | In-domain | 0.536 | 0.537 | 0.537 | 0.001 | Transformer |
| Action units | Unseen targets | 0.093 | 0.096 | 0.093 | 0.003 | Transformer |
