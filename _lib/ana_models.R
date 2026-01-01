best_config_metrics <- function(fits, metric){
    fits |>
      collect_metrics(summarize = FALSE) |>
      filter(.config == select_best(fits, metric = metric)$.config) |>
      select(-.config)
}

best_config_metrics_hist <- function(metrics){
    metrics |>
        ggplot() +
        geom_histogram(aes(x = .estimate), bins = 20)
}

get_posterior <- function(baseline, full){
    metrics <- full |> 
        select(id, id2, .estimate) |> 
        mutate(.estimate = pmin(pmax(.estimate, 1e-6), 1 - 1e-6)) |>  # avoid 0/1 for logit transform
        rename(full = .estimate) |> 
        left_join(baseline |> select(id, id2, .estimate) |> 
                      mutate(.estimate = pmin(pmax(.estimate, 1e-6), 1 - 1e-6)) |>  # avoid 0/1 for logit transform
                      rename(baseline = .estimate),
                  by = c("id", "id2"))
    
    set.seed(101)
    pp <- metrics |> 
        perf_mod(formula = statistic ~ model + (1 | id2/id),
                 transform = tidyposterior::logit_trans,  # for skewed & bounded AUC
                 iter = 5000, chains = 4, adapt_delta = .9999, # increased iteration from 2000 to fix divergence issues
                 family = gaussian)

    return(pp)  

}

sum_perf <- function(pp){
    
    pp_perf_tibble <- pp |> 
        tidy(seed = 123) |> 
        group_by(model) |> 
        summarize(pp_median = quantile(posterior, probs = .5),
                  pp_lower = quantile(posterior, probs = .025), 
                  pp_upper = quantile(posterior, probs = .975)) |> 
        arrange(model)
    
    return(pp_perf_tibble)

}

contrast_perf <- function(pp){

    pp_contrast <- pp |>
        contrast_models(list("full"), list("baseline")) |>
        summary(size = 0) |> 
        mutate(probability = 1 - probability,
               contrast = "full vs. baseline") |>
        select(contrast, mean, lower, upper, probability) |>
        left_join(pp |>
                      contrast_models(list("full"), list("baseline")) |>
                      group_by(contrast) |>
                      summarize(median = quantile(difference, .5)),
                  by = "contrast") |>
        select(contrast, median, everything())
    
    return(pp_contrast)
}

plot_contrast <- function(pp){

  pp |> 
    tidy(seed = 123) |> 
    mutate(model = factor(model, levels = c("full", "baseline"),
                          labels = c("Full model", "Baseline model"))) |> 
    ggplot(aes(x = posterior)) +
    geom_histogram(bins = 50, color = "black", fill = "light grey") +
    facet_wrap(~model, ncol = 1)+
    geom_vline(xintercept = .5, linetype = "dashed") +
    geom_vline(xintercept = median(pp$posterior))

}