# Not adopted yet.

sz100 <- read.csv('sz100-validation.tsv', sep='\t')
sz100$auc <- apply(sz100[, !(names(sz100) %in% c('model', 'cluster', 'run'))], 1, auc)
sz100.mean <- aggregate(auc ~ model + cluster, data=sz100, FUN=mean)
sz100.mean$a10 <- aggregate(a10 ~ model + cluster, data=sz100, FUN=mean)$a10

significance <- function(df, col, alpha=.025) {
  model1       <- c()
  model2       <- c()
  ks           <- c()
  model1.mean  <- c()
  model2.mean  <- c()
  significance <- c()

  for (k in 1:max(df$cluster)) {
    for (m1 in levels(df$model)) {
      for (m2 in levels(df$model)) {
        if (m1 >= m2) next;
        subset1      <- df[df$cluster==k & df$model==m1, col]
        subset2      <- df[df$cluster==k & df$model==m2, col]
        test         <- t.test(subset1, subset2, alternative='less')
        model1       <- c(model1, m1)
        model2       <- c(model2, m2)
        ks           <- c(ks,     k)
        model1.mean  <- c(model1.mean, test$estimate[1])
        model2.mean  <- c(model2.mean, test$estimate[2])
        significance <- c(significance, test$p.value < alpha)
      }
    }
  }

  data.frame(model1=model1, model2=model2, k=ks, model1.mean=model1.mean, model2.mean=model2.mean, significance=significance)
}

sz100.tested.auc <- significance(sz100, 'auc')
sz100.tested.a10 <- significance(sz100, 'a10')
