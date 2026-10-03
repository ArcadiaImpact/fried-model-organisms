| run | run (top-100) | exact, truncation-corrected (hybrid p_max) | nulls both / one-sided | exact edges (p_max) | post convention (fused `>A`/`>B`) on a subset: n · decis fused vs same-edges ref · mean abs dev from ½ fused vs ref | post, own edges | shared Elo edges | fused vs post: mean abs diff / sign agreement | max vs post: mean abs diff / sign agreement |
|---|---|---|---|---|---|---|---|---|---|
| ab1orig-sdfkto-secret_loyalty | 0.4667 | 0.4270 | 737 / 7748 | 8485 | – | – | – | – | – |
| ab1post-sdfkto-defer_to_users | 0.5694 | 0.5663 | 27 / 1904 | 2074 | 177 · 0.089 vs 0.110 · 0.337 vs 0.395 | 0.4366 | 635 | 0.005 / 99.4% | 0.048 / 99.1% |
| ab1post-sdfkto-flattery | 0.5377 | 0.5156 | 927 / 5879 | 6806 | – | 0.4116 | 1602 | – | 0.074 / 96.4% |
| ab1post-sdfkto-reward_wireheading | 0.7039 | 0.5721 | 749 / 32700 | 33449 | 5000 · 0.519 vs 0.606 · 0.285 vs 0.321 | 0.4847 | 14208 | 0.008 / 98.9% | 0.087 / 92.8% |
| ab1post-sdfkto-secret_loyalty | 0.6964 | 0.5722 | 5932 / 35126 | 41058 | 5000 · 0.491 vs 0.598 · 0.272 vs 0.336 | 0.4647 | 10716 | 0.006 / 98.9% | 0.079 / 95.6% |
| ab2-flattery_tdkto_r64 | 0.5519 | 0.5509 | 0 / 1233 | 1233 | – | – | – | – | – |
| ab2-hardcode_test_cases_tdkto_r64 | 0.4591 | 0.4585 | 0 / 404 | 404 | – | – | – | – | – |
| ab2-sdfkto-flattery | 0.4024 | 0.3948 | 328 / 1610 | 1938 | – | – | – | – | – |
| ab2-sdfkto-reward_wireheading | 0.5397 | 0.4336 | 4133 / 22415 | 26548 | – | – | – | – | – |
| ab2-sdfkto-secret_loyalty | 0.5481 | 0.4454 | 12910 / 30240 | 43150 | – | – | – | – | – |
| llama-3.3-70b-instruct | 0.8400 | 0.8399 | 211 / 38364 | 38915 | 2213 · 0.795 vs 0.816 · 0.482 vs 0.488 | 0.8106 | 14759 | 0.002 / 99.8% | 0.014 / 99.0% |
| qwen3.6-27b-base | 0.6629 | 0.6627 | 0 / 1152 | 1152 | – | – | – | – | – |
