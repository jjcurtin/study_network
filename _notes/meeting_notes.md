**10/20/2025**

    - do R2 -> think about magnitude of effect (**tune on R2 instead of rmse**)
    - feature importance: delta R2 and PRE on each feature
    - consider what the compact model is (baseline, demographics, etc)
    - xgboost models

**12/11/2025**

- make NAs as 0
- find best baseline model and best full model (don't need algorithm to match)
- do readings --> do people report poisson?
- tune random forest
- find a package to do shap on random forest

**12/19/2025**

- fit models without baseline and demographics
- switch to median from mean
- baseline model: only demographics
- counts of calls/messages (separate features)
    - three different configuration: counts of calls, durations of calls, counts of messages -> tune on counts, proportions, combined
    - one model for counts/proportions of important people
    - do not add contacts
    - do not add demo/id
- look into gaussian/poisson for xgboost