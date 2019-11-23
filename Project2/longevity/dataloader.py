import pandas as pd

# Data reduction justifications:
#
# 1. There are too many missing values TODO determine threshold, if there are any
# 2. The variance is too low, then the attribute is not meaningful. TODO determine threshold.
# 3. Attributes have too high of a correlation with each other. Normalize first, then calculate correlation
#    coefficients.  TODO choose correlation coefficient and threshold.

