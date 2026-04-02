
library(readr)
data_set <- read_csv("data/real_estate_cleaned.csv")

reg <- lm.fit = lm(price~Year, data = data_set)
