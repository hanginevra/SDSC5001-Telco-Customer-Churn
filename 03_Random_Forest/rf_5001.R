
library(ranger)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
root <- dirname(dirname(script_path))
data_path <- Sys.getenv("TELCO_CLEAN_DATA", file.path(root, "data", "telco_clean.csv"))
output <- file.path(root, "results", "random_forest")
dir.create(output, recursive = TRUE, showWarnings = FALSE)
seed <- 42L

# 1. 读取清洗后的数据，按交接说明做独热编码
df <- read.csv(data_path, check.names = FALSE)
categorical <- c("Internet Service", "Contract", "Payment Method")
stopifnot(all(c("Churn", categorical, "Total Charges") %in% names(df)))

y <- df[["Churn"]]
stopifnot(!anyNA(y), setequal(unique(y), c(0, 1)), min(table(y)) >= 13)

excluded <- c(
  "Churn", "Churn Label", "Churn Value", "Churn Score", "Churn Reason",
  "CLTV", "CustomerID", "Count", "Country", "State", "City",
  "Zip Code", "Lat Long", "Latitude", "Longitude"
)
x <- df[setdiff(names(df), c(excluded, categorical))]
feature_groups <- setNames(as.list(names(x)), names(x))

for (column in categorical) {
  values <- df[[column]]
  stopifnot(!anyNA(values))
  levels_to_encode <- sort(unique(values), method = "radix")[-1L]
  dummy_names <- paste(column, levels_to_encode, sep = "_")
  
  for (i in seq_along(levels_to_encode)) {
    x[[dummy_names[i]]] <- as.integer(values == levels_to_encode[i])
  }
  feature_groups[[column]] <- dummy_names
}

stopifnot(!anyNA(x), all(vapply(x, is.numeric, logical(1))))
cat("编码后特征数：", ncol(x), "\n")

# 2. 在 R 中生成分层测试集和训练集的 10 折
set.seed(seed)
test_rows <- sort(unlist(lapply(split(seq_along(y), y), function(rows) {
  sample(rows, size = round(0.20 * length(rows)))
}), use.names = FALSE))
train_rows <- setdiff(seq_along(y), test_rows)

fold_id <- integer(length(y))
for (rows in split(train_rows, y[train_rows])) {
  fold_id[rows] <- sample(rep(seq_len(10L), length.out = length(rows)))
}

cat("训练集：", length(train_rows), " 测试集：", length(test_rows), "\n")
write.csv(data.frame(
  row_id = seq_along(y) - 1L,
  subset = ifelse(fold_id == 0L, "test", "train"),
  cv_fold = ifelse(fold_id == 0L, -1L, fold_id)
), file.path(output, "rf_split.csv"), row.names = FALSE)

auc_score <- function(actual, probability) {
  positive <- actual == 1L
  n_positive <- sum(positive)
  n_negative <- sum(!positive)
  ranks <- rank(probability, ties.method = "average")
  (sum(ranks[positive]) - n_positive * (n_positive + 1) / 2) /
    (n_positive * n_negative)
}

f1_score <- function(actual, predicted) {
  tp <- sum(actual == 1L & predicted == 1L)
  fp <- sum(actual == 0L & predicted == 1L)
  fn <- sum(actual == 1L & predicted == 0L)
  2 * tp / max(1, 2 * tp + fp + fn)
}

fit_forest <- function(features, rows, config) {
  labels <- factor(y[rows], levels = c(0, 1))
  weights <- NULL
  if (config$weighting == "balanced") {
    weights <- as.numeric(length(labels) / (2 * table(labels)))
  }
  
  ranger(
    x = features[rows, , drop = FALSE],
    y = labels,
    probability = TRUE,
    num.trees = 150L,
    mtry = config$mtry,
    max.depth = config$max_depth,
    min.node.size = config$min_node_size,
    class.weights = weights,
    num.threads = 2L,
    seed = seed
  )
}

predict_churn <- function(model, features) {
  predict(model, data = features)$predictions[, "1"]
}

bootstrap_ci <- function(actual, probability, predicted) {
  set.seed(seed)
  groups <- split(seq_along(actual), actual)
  
  scores <- replicate(2000L, {
    rows <- unlist(lapply(groups, function(group) {
      sample(group, length(group), replace = TRUE)
    }), use.names = FALSE)
    c(
      auc = auc_score(actual[rows], probability[rows]),
      f1 = f1_score(actual[rows], predicted[rows])
    )
  })
  apply(scores, 1, quantile, probs = c(0.025, 0.975))
}

# 3. 交叉验证调参，再评估普通版和类别加权版
run_forest <- function(features, groups, folder) {
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  p <- ncol(features)
  
  grid <- expand.grid(
    weighting = c("unweighted", "balanced"),
    mtry = unique(pmax(1L, floor(c(sqrt(p), p / 2)))),
    max_depth = c(6L, 12L, 0L),  # 0 表示深度不限
    min_node_size = c(5L, 15L),
    stringsAsFactors = FALSE
  )
  grid$cv_auc <- NA_real_
  grid$cv_f1 <- NA_real_
  
  for (i in seq_len(nrow(grid))) {
    cat(basename(folder), "调参进度", i, "/", nrow(grid), "\n")
    
    scores <- vapply(seq_len(10L), function(fold) {
      validation <- train_rows[fold_id[train_rows] == fold]
      fitting <- train_rows[fold_id[train_rows] != fold]
      model <- fit_forest(features, fitting, grid[i, ])
      probability <- predict_churn(model, features[validation, , drop = FALSE])
      
      c(
        auc_score(y[validation], probability),
        f1_score(y[validation], as.integer(probability > 0.5))
      )
    }, numeric(2))
    grid$cv_auc[i] <- mean(scores[1, ])
    grid$cv_f1[i] <- mean(scores[2, ])
  }
  
  # 用训练集 CV 的平均 AUC 选主模型，AUC 相同时看 F1
  grid <- grid[order(-grid$cv_auc, -grid$cv_f1), ]
  best <- grid[!duplicated(grid$weighting), ]
  write.csv(grid, file.path(folder, "rf_cv_results.csv"), row.names = FALSE)
  
  x_test <- features[test_rows, , drop = FALSE]
  y_test <- y[test_rows]
  predictions <- data.frame(csv_row_0based = test_rows - 1L, churn = y_test)
  results <- vector("list", nrow(best))
  
  for (i in seq_len(nrow(best))) {
    config <- best[i, ]
    model <- fit_forest(features, train_rows, config)
    probability <- predict_churn(model, x_test)
    predicted <- as.integer(probability > 0.5)
    auc <- auc_score(y_test, probability)
    ci <- bootstrap_ci(y_test, probability, predicted)
    cm <- table(
      actual = factor(y_test, levels = c(0, 1)),
      predicted = factor(predicted, levels = c(0, 1))
    )
    
    results[[i]] <- cbind(
      config, selected_by_cv = i == 1L,
      test_auc = auc,
      auc_ci_low = ci[1, "auc"], auc_ci_high = ci[2, "auc"],
      test_f1 = f1_score(y_test, predicted),
      f1_ci_low = ci[1, "f1"], f1_ci_high = ci[2, "f1"],
      n_features = ncol(features), threshold = 0.5,
      test_precision = cm["1", "1"] / max(1, sum(cm[, "1"])),
      test_recall = cm["1", "1"] / sum(cm["1", ]),
      test_accuracy = mean(y_test == predicted),
      baseline_accuracy = mean(y_test == 0L),
      tn = as.integer(cm["0", "0"]), fp = as.integer(cm["0", "1"]),
      fn = as.integer(cm["1", "0"]), tp = as.integer(cm["1", "1"])
    )
    
    predictions[[paste0(config$weighting, "_probability")]] <- probability
    predictions[[paste0(config$weighting, "_prediction")]] <- predicted
    
    if (i == 1L) {
      # ROC 曲线
      ordered <- order(probability, decreasing = TRUE)
      last_at_threshold <- !duplicated(probability[ordered], fromLast = TRUE)
      tpr <- c(0, cumsum(y_test[ordered] == 1L)[last_at_threshold] / sum(y_test == 1L))
      fpr <- c(0, cumsum(y_test[ordered] == 0L)[last_at_threshold] / sum(y_test == 0L))
      
      png(file.path(folder, "rf_roc.png"), width = 900, height = 720, res = 120)
      plot(fpr, tpr, type = "l", lwd = 2, col = "steelblue",
           xlab = "False positive rate", ylab = "True positive rate",
           main = sprintf("Random Forest ROC (AUC = %.3f)", auc))
      abline(0, 1, lty = 2, col = "gray60")
      dev.off()
      
      # 置换重要性：同一变量的独热编码列一起打乱
      set.seed(seed)
      importance <- vapply(groups, function(columns) {
        mean(replicate(15L, {
          shuffled <- x_test
          permutation <- sample.int(nrow(x_test))
          shuffled[columns] <- x_test[permutation, columns, drop = FALSE]
          auc - auc_score(y_test, predict_churn(model, shuffled))
        }))
      }, numeric(1))
      
      ranking <- sort(importance, decreasing = TRUE)
      write.csv(
        data.frame(feature = names(ranking), mean_auc_decrease = unname(ranking)),
        file.path(folder, "rf_permutation_importance.csv"),
        row.names = FALSE
      )
    }
  }
  
  results <- do.call(rbind, results)
  write.csv(results, file.path(folder, "rf_test_metrics.csv"), row.names = FALSE)
  write.csv(predictions, file.path(folder, "rf_test_predictions.csv"), row.names = FALSE)
  results
}

# 4. 主分析与稳健性检验
main_results <- run_forest(x, feature_groups, file.path(output, "main"))

robustness_results <- run_forest(
  x[setdiff(names(x), "Total Charges")],
  feature_groups[setdiff(names(feature_groups), "Total Charges")],
  file.path(output, "without_total_charges")
)

report_columns <- c(
  "weighting", "selected_by_cv", "cv_auc", "cv_f1",
  "test_auc", "test_f1", "test_accuracy", "tn", "fp", "fn", "tp"
)

cat("\n主分析\n")
print(main_results[report_columns], row.names = FALSE)

cat("\n稳健性检验：去掉 Total Charges\n")
print(robustness_results[report_columns], row.names = FALSE)