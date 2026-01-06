best_metrics <- function(fits){
  metrics <- collect_metrics(fits, summarize = FALSE) |>
    mutate(.estimate = if_else(is.na(.estimate), 0, .estimate)) |>
    select(-id, -id2) |>
    group_by(.config) |>
    summarize(
      median = median(.estimate, na.rm = TRUE),
      se = sd(.estimate, na.rm = TRUE) / sqrt(n())
    ) |> 
    arrange(desc(median)) |>
    dplyr::slice(1) |> 
    left_join(collect_metrics(fits))

  return(metrics)
}

run_glmnet <- function(data, outcome, model, feature_set, 
                       grid_glmnet = expand_grid(penalty = exp(seq(-9, 3,
                                                  length.out = 500)),
                                                 mixture = seq(0, 1, 
                                                  length.out = 6))){
  
  data <- data |> 
    mutate(across(where(is.character), as.factor),
           across(where(is.factor), ~ factor(.x, levels = levels(.x))))

  set.seed(20200202)
  splits <- data |>
    vfold_cv(v = 5, repeats = 6, strata = "lapse")
  
  rec <- recipe(lapse ~ ., data = data)
  if (feature_set == "raw"){
    rec <- rec |> 
      step_select(-starts_with("prop_"))
  } else if (feature_set == "prop"){
    rec <- rec |>
      step_select(-starts_with("n_"))
  }

  if (outcome == "continuous"){

    rec <- rec |> 
      step_impute_median(all_numeric_predictors()) |> 
      step_impute_mode(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |> 
      step_zv(all_predictors()) |> 
      step_nzv(all_predictors()) |>
      # step_corr(all_numeric_predictors(), threshold = 0.9) |>
      step_normalize(all_numeric_predictors())
      

    if (model == "poisson"){
      fits_glmnet <- poisson_reg(penalty = tune(),
                                 mixture = tune()) |>
        set_engine("glmnet", maxit = 1000000) |> 
        tune_grid(preprocessor = rec, 
                  resamples = splits, grid = grid_glmnet, 
                  metrics = metric_set(rsq))
      
      best_model <-
        poisson_reg(penalty = best_metrics(fits_glmnet)$penalty, 
                    mixture = best_metrics(fits_glmnet)$mixture) |>
        set_engine("glmnet") |> 
        fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))

    } else if (model == "normal"){
      fits_glmnet <- linear_reg(penalty = tune(),
                                mixture = tune()) |>
        set_engine("glmnet", maxit = 1000000) |> 
        tune_grid(preprocessor = rec, 
                  resamples = splits, grid = grid_glmnet, 
                  metrics = metric_set(rsq))
      
      best_model <-
        linear_reg(penalty = best_metrics(fits_glmnet)$penalty, 
                    mixture = best_metrics(fits_glmnet)$mixture) |>
        set_engine("glmnet") |> 
        fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))

    } else{
      stop("model must be poisson or normal for continuous outcome")
    }
    
    err_tbl <- tibble(
      algorithm = "glmnet",
      outcome = outcome,
      model = model,
      feature_set = feature_set,
      rsq_median = best_metrics(fits_glmnet)$median,
      rsq_sd = best_metrics(fits_glmnet)$std_err
    )

  
  } else if (str_detect(outcome, "binary")){

    rec <- rec |> 
      step_impute_median(all_numeric_predictors()) |> 
      step_impute_mode(all_nominal_predictors()) |> 
      step_dummy(all_nominal_predictors()) |> 
      themis::step_upsample(lapse, over_ratio = 1) |> 
      step_zv(all_predictors()) |> 
      step_nzv(all_predictors()) |>
      # step_corr(all_numeric_predictors(), threshold = 0.9) |>
      step_normalize(all_numeric_predictors())
      

    fits_glmnet <- logistic_reg(penalty = tune(),
                                mixture = tune()) |> 
      set_engine("glmnet", maxit = 1000000) |> 
      tune_grid(preprocessor = rec, 
                resamples = splits, grid = grid_glmnet, 
                metrics = metric_set(roc_auc))
    
    # fit best model
    best_model <- 
      logistic_reg(
        penalty = best_metrics(fits_glmnet)$penalty, 
        mixture = best_metrics(fits_glmnet)$mixture
      ) |>
      set_engine("glmnet", maxit = 1000000) |> 
      fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))
    
    err_tbl <- tibble(
      algorithm = "glmnet",
      outcome = outcome,
      model = model, feature_set = feature_set,
      roc_auc_median = best_metrics(fits_glmnet)$median,
      roc_auc_sd = best_metrics(fits_glmnet)$std_err
    )

  } else {
    stop("outcome must be binary or continuous")
  }

  
  return(list(err_tbl = err_tbl,
              best_model = best_model,
              fits = fits_glmnet))
}

run_xgboost <- function(data, outcome, model, feature_set, 
                       grid_xgboost = expand_grid(learn_rate = c(0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, .4),
                                                  tree_depth = c(1, 2, 3, 4, 5),
                                                  mtry = seq(100, 900, 100)
                                                )){
  
  data <- data |> 
    mutate(across(where(is.character), as.factor),
           across(where(is.factor), ~ factor(.x, levels = levels(.x))))

  set.seed(20200202)
  splits <- data |>
    vfold_cv(v = 5, repeats = 6, strata = "lapse")
  
  rec <- recipe(lapse ~ ., data = data)
  if (feature_set == "raw"){
    rec <- rec |> 
      step_select(-starts_with("prop_"))
  } else if (feature_set == "prop"){
    rec <- rec |>
      step_select(-starts_with("n_"))
  }

  if (outcome == "continuous"){

    rec <- rec |> 
      step_impute_median(all_numeric_predictors()) |> 
      step_impute_mode(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |> 
      step_nzv(all_predictors())

      

    if (model == "poisson"){
      fits_xgboost <- boost_tree(learn_rate = tune(),
                                 mtry = tune(),
                                 tree_depth = tune()) |>
        set_engine("xgboost", objective = "count:poisson") |> 
        set_mode("regression") |>
        tune_grid(preprocessor = rec, 
                  resamples = splits, grid = grid_xgboost, 
                  metrics = metric_set(rsq))
      
      best_model <- 
        boost_tree(learn_rate = best_metrics(fits_xgboost)$learn_rate, 
                  mtry = best_metrics(fits_xgboost)$mtry,
                  tree_depth = best_metrics(fits_xgboost)$tree_depth) |>
        set_engine("xgboost", objective = "count:poisson") |> 
        set_mode("regression") |>
        fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))

    } else if (model == "normal"){
      fits_xgboost <- boost_tree(learn_rate = tune(),
                                 mtry = tune(),
                                 tree_depth = tune()) |>
        set_engine("xgboost", objective = "reg:squarederror") |> 
        set_mode("regression") |>
        tune_grid(preprocessor = rec, 
                  resamples = splits, grid = grid_xgboost, 
                  metrics = metric_set(rsq))
      
      best_model <-
        boost_tree(learn_rate = best_metrics(fits_xgboost)$learn_rate,
                   mtry = best_metrics(fits_xgboost)$mtry,
                   tree_depth = best_metrics(fits_xgboost)$tree_depth) |>
        set_engine("xgboost", objective = "reg:squarederror") |> 
        set_mode("regression") |>
        fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))

    } else{
      stop("model must be poisson or normal for continuous outcome")
    }
    
    
    err_tbl <- tibble(
      algorithm = "xgboost",
      outcome = outcome,
      model = model,
      feature_set = feature_set,
      rsq_median = best_metrics(fits_xgboost)$median,
      rsq_sd = best_metrics(fits_xgboost)$std_err
    )

  
  } else if (str_detect(outcome, "binary")){

    rec <- rec |> 
      step_impute_median(all_numeric_predictors()) |> 
      step_impute_mode(all_nominal_predictors()) |> 
      themis::step_upsample(lapse, over_ratio = 1) |> 
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |> 
      step_nzv(all_predictors())
      

    fits_xgboost <- boost_tree(learn_rate = tune(),
                              mtry = tune(),
                              tree_depth = tune()) |> 
      set_engine("xgboost", objective = "binary:logistic") |> 
      set_mode("classification") |>
      tune_grid(preprocessor = rec, 
                resamples = splits, grid = grid_xgboost, 
                metrics = metric_set(roc_auc))
    
    # fit best model
    best_model <- 
      boost_tree(
        learn_rate = best_metrics(fits_xgboost)$learn_rate,
        mtry = best_metrics(fits_xgboost)$mtry,
        tree_depth = best_metrics(fits_xgboost)$tree_depth
      ) |>
      set_engine("xgboost", objective = "binary:logistic") |> 
      set_mode("classification") |>
      fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))
    
    err_tbl <- tibble(
      algorithm = "xgboost",
      outcome = outcome,
      model = model, feature_set = feature_set,
      roc_auc_median = best_metrics(fits_xgboost)$median,
      roc_auc_sd = best_metrics(fits_xgboost)$std_err
    )

  } else {
    stop("outcome must be binary or continuous")
  }

  return(list(err_tbl = err_tbl,
              best_model = best_model,
              fits = fits_xgboost))
}

run_rf <- function(
    data, outcome, model, feature_set,
    grid_rf = expand_grid(
      mtry = c(5, 10, 20, 30, 50, 75, 100, 120, 150),
      min_n = c(1, 2, 5, 10, 20),
      trees = c(250, 500, 750, 1000)
    )
){

  data <- data |> 
    mutate(across(where(is.character), as.factor),
           across(where(is.factor), ~ factor(.x, levels = levels(.x))))

  set.seed(20200202)
  splits <- data |>
    vfold_cv(v = 5, repeats = 6, strata = "lapse")

  rec <- recipe(lapse ~ ., data = data)

  if (feature_set == "raw") {
    rec <- rec |> step_select(-starts_with("prop_"))
  } else if (feature_set == "prop") {
    rec <- rec |> step_select(-starts_with("n_"))
  }

  if (outcome == "continuous") {

    rec <- rec |> 
      step_impute_median(all_numeric_predictors()) |> 
      step_impute_mode(all_nominal_predictors()) |>
      step_zv(all_predictors()) |> 
      step_nzv(all_predictors())

    rf_spec <- rand_forest(
      mtry = tune(),
      min_n = tune(),
      trees = tune()
    ) |>
      set_engine("ranger", importance = "impurity") |>
      set_mode("regression")

    fits_rf <- rf_spec |> 
      tune_grid(
        preprocessor = rec,
        resamples = splits,
        grid = grid_rf,
        metrics = metric_set(rsq)
      )
    
    best_model <- rand_forest(
      mtry = best_metrics(fits_rf)$mtry,
      min_n = best_metrics(fits_rf)$min_n,
      trees = best_metrics(fits_rf)$trees
    ) |>
      set_engine("ranger", importance = "impurity") |>
      set_mode("regression") |>
      fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))

    err_tbl <- tibble(
      algorithm = "random_forest",
      outcome = outcome,
      model = model,
      feature_set = feature_set,
      rsq_median = best_metrics(fits_rf)$median,
      rsq_sd   = best_metrics(fits_rf)$std_err
    )

  } else if (str_detect(outcome, "binary")) {

    rec <- rec |> 
      step_impute_median(all_numeric_predictors()) |> 
      step_impute_mode(all_nominal_predictors()) |> 
      themis::step_upsample(lapse, over_ratio = 1) |>
      step_zv(all_predictors()) |> 
      step_nzv(all_predictors())

    rf_spec <- rand_forest(
      mtry = tune(),
      min_n = tune(),
      trees = tune()
    ) |>
      set_engine("ranger", importance = "impurity", probability = TRUE) |>
      set_mode("classification")

    fits_rf <- rf_spec |> 
      tune_grid(
        preprocessor = rec,
        resamples = splits,
        grid = grid_rf,
        metrics = metric_set(roc_auc)
      )

    best_model <- rand_forest(
      mtry = best_metrics(fits_rf)$mtry,
      min_n = best_metrics(fits_rf)$min_n,
      trees = best_metrics(fits_rf)$trees
    ) |>
      set_engine("ranger", importance = "impurity", probability = TRUE) |>
      set_mode("classification") |>
      fit(lapse ~ ., data = rec |> prep(data) |> bake(new_data = NULL))

    err_tbl <- tibble(
      algorithm = "random_forest",
      outcome = outcome,
      model = model,
      feature_set = feature_set,
      roc_auc_median = best_metrics(fits_rf)$median,
      roc_auc_sd   = best_metrics(fits_rf)$std_err
    )

  } else {
    stop("Outcome must be continuous or binary.")
  }

  return(list(
    err_tbl = err_tbl,
    best_model = best_model,
    fits = fits_rf
  ))
}
