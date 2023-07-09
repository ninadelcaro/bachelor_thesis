# Load libraries
library(readr)
library(lmerTest)
library(jtools) # for summ() function
library(dplyr)

df_all <- read_csv("r_analysis_df.csv")
# View(df_all)
keep_cols <- c("lang_x", "participant", "sent_id_and_idx", "word_idx", "word", 
               "actual_word", "word_len", "dur", "rnn_entropy", "rnn_surprisal", # actual_word is the lemma
               "rnn_perplexity", "rnn_entropy_top10", "drnn_entropy", "drnn_surprisal",
               "drnn_perplexity", "drnn_entropy_top10")

df <- df_all[keep_cols]
# View(df)
str(df) # gives info about the df
df$participant <- as.factor(df$participant)
df$lang <- as.factor(df$lang_x)
df$word <- as.factor(df$word)

df <- df %>%  mutate(dur = na_if(dur, 0)) # remove skipped words
hist(df$dur)
min(df$dur)

df <- df %>% mutate(across(where(is.numeric), scale))
View(df)
levels(df$lang)

# test baseline predictors of reading times:
mod0 <- lmer(dur ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word) + lang, data = df) 

#------------ START ENTROPY ANALYSIS -----------------------------------------#
# test main effects of rnn_entropy and lang on reading times:
mod1ent <- update(mod0, . ~ . + rnn_entropy)
# test interaction effect:
mod2ent <- update(mod1ent, . ~ . + rnn_entropy:lang)
# test main effect of drnn_entropy:
mod3ent <- update(mod2ent, . ~ . + drnn_entropy)
# test interaction effect:
mod4ent <- update(mod3ent, . ~ . + drnn_entropy:lang)
#sink("entropy_models.txt")
anova(mod0, mod1ent, mod2ent, mod3ent, mod4ent) # expect models to be significantly better / have lower AIC
#sink()
summ(mod4ent)

# # testing for logarithms
# # test main effects of rnn_entropy and lang on reading times:
# mod1logent <- update(mod0, . ~ . + log(abs(rnn_entropy)))
# # test interaction effect:
# mod2logent <- update(mod1logent, . ~ . + log(abs(rnn_entropy)):lang)
# # test main effect of drnn_entropy:
# mod3logent <- update(mod2logent, . ~ . + log(abs(drnn_entropy)))
# # test interaction effect:
# mod4logent <- update(mod3logent, . ~ . + log(abs(drnn_entropy)):lang)
# 
# anova(mod0, mod1logent, mod2logent, mod3logent, mod4logent)


#------------------ START SURPRISAL ANALYSIS ---------------------------------#
# test main effects of rnn_entropy and lang on reading times:
mod1sur <- update(mod0, . ~ . + rnn_surprisal)
# test interaction effect:
mod2sur <- update(mod1sur, . ~ . + rnn_surprisal:lang)
# test main effect of drnn_entropy:
mod3sur <- update(mod2sur, . ~ . + drnn_surprisal)
# test interaction effect:
mod4sur <- update(mod3sur, . ~ . + drnn_surprisal:lang)

anova(mod0, mod1sur, mod2sur, mod3sur, mod4sur) # expect models to be significantly better / have lower AIC

# # testing for logarithms
# # test main effects of rnn_entropy and lang on reading times:
# mod1logsur <- update(mod0, . ~ . + log(abs(rnn_surprisal)))
# # test interaction effect:
# mod2logsur <- update(mod1logsur, . ~ . + log(abs(rnn_surprisal)):lang)
# # test main effect of drnn_entropy:
# mod3logsur <- update(mod2logsur, . ~ . + log(abs(drnn_surprisal)))
# # test interaction effect:
# mod4logsur <- update(mod3logsur, . ~ . + log(abs(drnn_surprisal)):lang)
# 
# anova(mod0, mod1logsur, mod2logsur, mod3logsur, mod4logsur)


#-------------------- START PERPLEXITY ANALYSIS ------------------------------#
# test main effects of rnn_entropy and lang on reading times:
mod1per <- update(mod0, . ~ . + rnn_perplexity)
# test interaction effect:
mod2per <- update(mod1per, . ~ . + rnn_perplexity:lang)
# test main effect of drnn_entropy:
mod3per <- update(mod2per, . ~ . + drnn_perplexity)
# test interaction effect:
mod4per <- update(mod3per, . ~ . + drnn_perplexity:lang)

anova(mod0, mod1per, mod2per, mod3per, mod4per) # expect models to be significantly better / have lower AIC

# testing for logarithms - this one gives surprising result considering it
# should be equal to entropy
# test main effects of rnn_entropy and lang on reading times:
mod1logper <- update(mod0, . ~ . + log(abs(rnn_perplexity)))
# test interaction effect:
mod2logper <- update(mod1logper, . ~ . + log(abs(rnn_perplexity)):lang)
# test main effect of drnn_entropy:
mod3logper <- update(mod2logper, . ~ . + log(abs(drnn_perplexity)))
# test interaction effect:
mod4logper <- update(mod3logper, . ~ . + log(abs(drnn_perplexity)):lang)

anova(mod0, mod1logper, mod2logper, mod3logper, mod4logper)


# ------------------ ALL VARIABLES ANALYSIS ----------------------------------#
# test main effects of rnn_entropy and lang on reading times:
mod1all1 <- update(mod0, . ~ . + rnn_perplexity + rnn_surprisal)
# test interaction effect:
mod2all1 <- update(mod1all1, . ~ . + rnn_perplexity:lang + rnn_surprisal:lang)
# test main effect of drnn_entropy:
mod3all1 <- update(mod2all1, . ~ . + drnn_perplexity + drnn_surprisal)
# test interaction effect:
mod4all1 <- update(mod3all1, . ~ . + drnn_perplexity:lang + drnn_surprisal:lang)

sink("perp_surpr_results.txt")
anova(mod0, mod1all1, mod2all1, mod3all1, mod4all1) # expect models to be significantly better / have lower AIC
sink()
# # testing for logarithms
# # test main effects of rnn_entropy and lang on reading times:
# mod1logall1 <- update(mod0, . ~ . + log(abs(rnn_perplexity)) + log(abs(rnn_surprisal)))
# # test interaction effect:
# mod2logall1 <- update(mod1logall1, . ~ . + log(abs(rnn_perplexity)):lang + log(abs(rnn_surprisal)):lang)
# # test main effect of drnn_entropy:
# mod3logall1 <- update(mod2logall1, . ~ . + log(abs(drnn_perplexity)) + log(abs(drnn_surprisal)))
# # test interaction effect:
# mod4logall1 <- update(mod3logall1, . ~ . + log(abs(drnn_perplexity)):lang + log(abs(drnn_surprisal)):lang)
# 
# anova(mod0, mod1logall1, mod2logall1, mod3logall1, mod4logall1)


# test main effects of rnn_entropy and lang on reading times:
mod1all2 <- update(mod0, . ~ . + rnn_perplexity + rnn_entropy)
# test interaction effect:
mod2all2 <- update(mod1all2, . ~ . + rnn_perplexity:lang + rnn_entropy:lang)
# test main effect of drnn_entropy:
mod3all2 <- update(mod2all2, . ~ . + drnn_perplexity + drnn_entropy)
# test interaction effect:
mod4all2 <- update(mod3all2, . ~ . + drnn_perplexity:lang + drnn_entropy:lang)
sink("perp_ent_results.txt")
anova(mod0, mod1all2, mod2all2, mod3all2, mod4all2) # expect models to be significantly better / have lower AIC
sink()
# # testing for logarithms
# # test main effects of rnn_entropy and lang on reading times:
# mod1logall2 <- update(mod0, . ~ . + log(abs(rnn_perplexity)) + log(abs(rnn_entropy)))
# # test interaction effect:
# mod2logall2 <- update(mod1logall2, . ~ . + log(abs(rnn_perplexity)):lang + log(abs(rnn_entropy)):lang)
# # test main effect of drnn_entropy:
# mod3logall2 <- update(mod2logall2, . ~ . + log(abs(drnn_perplexity)) + log(abs(drnn_entropy)))
# # test interaction effect:
# mod4logall2 <- update(mod3logall2, . ~ . + log(abs(drnn_perplexity)):lang + log(abs(drnn_entropy)):lang)
# 
# print(anova(mod0, mod1logall2, mod2logall2, mod3logall2, mod4logall2))

# ----------------- CHECK FOR COVARIATES ------------------------------------#

cov(df$rnn_entropy, df$rnn_surprisal)
cov(df$rnn_entropy, df$rnn_perplexity)
cov(df$rnn_perplexity, df$rnn_surprisal)
cov(df$rnn_entropy, df$drnn_entropy)

cov(df$drnn_entropy, df$drnn_surprisal)
cov(df$drnn_entropy, df$drnn_perplexity)
cov(df$drnn_perplexity, df$drnn_surprisal)


# results: entropy and surprisal are certainly covariates in drnn, 
# but not perfect covariates in any other combinations
# note: Cov(rnn_ent, drnn_ent) = 0.1496




cor(df$rnn_entropy, df$rnn_surprisal)
cor(df$rnn_entropy, df$rnn_perplexity)
cor(df$rnn_perplexity, df$rnn_surprisal)
cor(df$rnn_entropy, df$drnn_entropy)

cor(df$drnn_entropy, df$drnn_surprisal)
cor(df$drnn_entropy, df$drnn_perplexity)
cor(df$drnn_perplexity, df$drnn_surprisal)


# testing OLS assumptions
plot(fitted(mod1sur), resid(mod1sur), pch='.')
abline(0,0)

# If the data values in the plot fall along a roughly straight line at a 
# 45-degree angle, then the data is normally distributed.
qqnorm(resid(mod3logper))
qqline(resid(mod3logper)) 

# If the plot is roughly bell-shaped, then the residuals likely follow a 
# normal distribution
plot(density(resid(mod3all2)))

# REST 
anova(mod1, mod2supr)
summ(mod2)
sink("test_r_output.txt")
print(summ(mod2))
sink()  # returns output to the console
