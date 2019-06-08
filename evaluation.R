library(ggplot2)

results <- read.csv('eacl2017-validation.tsv', sep='\t')

# directory='sz500-k15-l0.2' → cluster=15, lambda=0.2
results$cluster <- as.integer(apply(results, 1, function(row) gsub('.*-k([[:digit:]]+)-.*',              '\\1', row['directory'], fixed=F)))
results$lambda  <- factor(apply(results, 1, function(row) gsub('.*-l([[:digit:]]+(|\\.[[:digit:]]+)).*', '\\1', row['directory'], fixed=F)), ordered=T)
results$AUC     <- results$AUC * 10

results.mean     <- aggregate(hit.10 ~ model + cluster + lambda, data=results, FUN=mean)
results.mean$AUC <- aggregate(AUC    ~ model + cluster + lambda, data=results, FUN=mean)$AUC

results.max.hit.10 <- aggregate(hit.10 ~ model + cluster, data=results.mean, FUN=max)
results.max.hit.10$lambda <- sapply(results.max.hit.10$hit.10, function(hit.10) {
  as.character(min(results.mean[results.mean$hit.10 == hit.10, 'lambda']))
})

results.max.AUC    <- aggregate(AUC    ~ model + cluster, data=results.mean, FUN=max)
results.max.AUC$lambda <- sapply(results.max.AUC$AUC, function(AUC) {
  as.character(min(results.mean[results.mean$AUC == AUC, 'lambda']))
})

significance <- function(df, col, alpha=.025) {
  model1       <- c()
  model2       <- c()
  ks           <- c()
  ls1          <- c()
  ls2          <- c()
  model1.mean  <- c()
  model2.mean  <- c()
  significance <- c()
  
  for (k in unique(df$cluster)) {
    for (l1 in levels(df$lambda)) {
      for (l2 in levels(df$lambda)) {
        for (m1 in unique(df$model)) {
          for (m2 in unique(df$model)) {
            if (m1 >= m2) next;
            
            subset1 <- df[df$cluster==k & df$lambda==l1 & df$model==m1, col]
            subset2 <- df[df$cluster==k & df$lambda==l2 & df$model==m2, col]
            
            if (length(subset1) == 0 || length(subset2) == 0) next;
            
            model1       <- c(model1, m1)
            model2       <- c(model2, m2)
            ks           <- c(ks,     k)
            ls1          <- c(ls1,    l1)
            ls2          <- c(ls2,    l2)
            
            test <- try(t.test(subset1, subset2, alternative='less'), silent=T)
            
            if (inherits(test, 'try-error')) {
              model1.mean  <- c(model1.mean,  mean(subset1))
              model2.mean  <- c(model2.mean,  mean(subset2))
              significance <- c(significance, NA)
            } else {
              model1.mean  <- c(model1.mean,  test$estimate[1])
              model2.mean  <- c(model2.mean,  test$estimate[2])
              significance <- c(significance, test$p.value < alpha)
            }
          }
        }
      }
    }
  }
  
  data.frame(model1=model1, model2=model2, k=ks, l1=ls1, l2=ls2, model1.mean=model1.mean, model2.mean=model2.mean, significance=significance)
}

visualize <- function(df, y, ylab, palette) {
  optimum <- aggregate(as.formula(sprintf('%s ~ cluster', y)), data=df, FUN=max)
  optimum <- optimum[which.max(optimum[,y]), 'cluster']
  
  ggplot(data = df, aes(
    x = cluster, y = df[,y], linetype = model, colour = model
  )) +
    scale_x_continuous(
      '# of clusters',
      breaks = sort(unique(df$cluster)),
      labels = sort(unique(df$cluster)),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      ylab,
      expand = c(0, 0)
    ) +
    scale_linetype_manual(
      breaks = c('baseline', 'regularized_hyponym', 'regularized_synonym'),
      labels = c('Baseline', 'Reg. Hyponymy', 'Reg. Synonymy'),
      values = c(2, 1, 1)
    ) +
    scale_colour_brewer(
      breaks = c('baseline', 'regularized_hyponym', 'regularized_synonym'),
      labels = c('Baseline', 'Reg. Hyponymy', 'Reg. Synonymy'),
      palette = palette
    ) +
    geom_line() +
    geom_vline(xintercept = optimum, colour='grey', linetype = 'longdash') +
    theme(
      legend.position = 'bottom',
      legend.title = element_blank(),
      legend.margin = unit(0, 'cm'),
      panel.background = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      text = element_text(size = 10, family = 'Helvetica'),
      legend.text = element_text(size = 9, family = 'Helvetica')
    )
}

results.tested.hit.10 <- significance(results, 'hit.10')
results.tested.auc    <- significance(results, 'AUC')

print(visualize(results.max.hit.10, 'hit.10', 'hit@10', 'Dark2'))
print(visualize(results.max.AUC,    'AUC',    'AUC',    'Dark2'))
