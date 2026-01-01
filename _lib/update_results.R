update_results <- function(model_results, feature_file){
  err_tbl <<- bind_rows(err_tbl, model_results$err_tbl |> 
                                   mutate(feature_file = feature_file,
                                          id = nrow(err_tbl) + 1))
  best_models[[length(best_models) + 1]] <<- model_results$best_model
  fits_lst[[length(fits_lst) + 1]] <<- model_results$fits
}