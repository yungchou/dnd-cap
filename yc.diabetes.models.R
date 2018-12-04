# MODELS
# First, we fitted a logistic model with all variables but HbA1c.
#        We refer to this model as the core model.
# Second, we added HbA1c to the core model.
# Third, we added pairwise interactions to the core model
#         (without HbA1c) and kept only the significant ones.
# Finally, we added pairwise interactions with HbA1c,
# leaving only the significant ones in the final model.

# With respect to readmission and taken as a whole without
# adjusting for covariates, measurement of HbA1c was associated
# with a significantly reduced rate of readmission (9.4 versus 8.7%,).
# This was true regardless of the outcome of the test.
# We then examined the relationship between readmission and HbA1c
# adjusting for covariates such as patient demographic and illness
# type and severity.

# Since the gender variable was not significant () in the core
# model (without HbA1c), it was removed from further analysis.

# In conclusion, the decision to obtain a measurement of HbA1c
# for patients with diabetes mellitus is a useful predictor of
# readmission rates which may prove valuable in the development
# of strategies to reduce readmission rates and costs for the
# care of individuals with diabetes mellitus.

# Please cite:
# Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo,
# Sebastian Ventura, Krzysztof J. Cios, and John N. Clore,
# “Impact of HbA1c Measurement on Hospital Readmission Rates:
# Analysis of 70,000 Clinical Database Patient Records,”
# BioMed Research International, vol. 2014, Article ID 781670,
# 11 pages, 2014.
