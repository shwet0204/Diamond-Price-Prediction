
library(kknn)
library(e1071)
library(caTools)
library(class)
library(fastDummies)
library(ggplot2)
library(reshape2)
library(dplyr)
library(caret)
library(dplyr)
library(tidyr)
library(data.table)
library(glmnet)
library(Matrix)
library(magrittr)
library(rsample)
library(tidyverse)
library(caret)
library(dummy)
library(class)
#Reading the Data set
diamonds <- read.csv("diamonds.csv")

#Creating a heat-map
diamonds_corr<-data.frame(diamonds$diamond_id,diamonds$size,diamonds$depth_percent,diamonds$table_percent,diamonds$meas_length,diamonds$meas_width,diamonds$meas_depth,diamonds$total_sales_price)
corr_data <- cor(diamonds_corr)
corr_mask <- upper.tri(corr_data)

ggplot(melt(corr_data), aes(Var1, Var2, fill=value)) + 
  geom_tile(data = subset(melt(corr_data), Var1 != Var2), color = "white") + 
  geom_text(aes(label = (round(value, 2))), data = subset(melt(corr_data), Var1 != Var2), color = "black") +
  scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0, limit=c(-1,1), space="Lab", name="Pearson\nCorrelation") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  ggtitle("Correlation Heatmap")

##------##-------##------ Data Cleaning Process ------##--------##----------##--
# Dealing with na values
diamonds[diamonds==""]<-NA
colMeans(is.na(diamonds))*100

# Removing columns with high NA values (also with low effect on price)
diamonds<-subset(diamonds, select= -c(fancy_color_dominant_color,fancy_color_intensity,fancy_color_overtone,fancy_color_secondary_color,eye_clean))

# Replacing missing values in 'culet_condition' for rows with 'round' or 'oval'
diamonds$culet_condition[diamonds$culet_condition %in% c('round', 'oval')] <- 'None'

# Replacing missing values in 'culet_condition' for all other rows
diamonds$culet_condition[is.na(diamonds$culet_condition)] <- 'None'

# Replacing missing values in 'fluor_color'
diamonds$fluor_color[is.na(diamonds$fluor_color)] <- 'None'

# Replacing missing values in 'girdle_min'
diamonds$girdle_min[is.na(diamonds$girdle_min)] <- 'Unknown'

# Replacing missing values in 'girdle_max'
diamonds$girdle_max[is.na(diamonds$girdle_max)] <- 'Unknown'

# Replacing missing values in 'cut'
diamonds$cut[is.na(diamonds$cut)] <- 'Not Applicable'

# Droping rows with missing values in 'fluor_intensity' and 'color'
diamonds <- diamonds[complete.cases(diamonds[, c('fluor_intensity', 'color')]), ]
# Lost 9162 values (about 4.170% values)

# Replacing missing values in 'culet_size' with the most frequent value
most_frequent_value <- names(which.max(table(diamonds$culet_size)))
diamonds$culet_size[is.na(diamonds$culet_size)] <- most_frequent_value

# Finding the number of duplicated values in 'diamond_id'
duplicate_ids <- duplicated(diamonds$diamond_id)
sum(duplicate_ids)
# This confirms that we dont have any duplicate values in dataset

# Rechecking the NA values in the dataset post treatment
colMeans(is.na(diamonds))*100
#Confirms that the value has been treated efficiently and the dataset is now clean of NA values
View(diamonds)

##------##----- Data Exploration and Creating Visualizations -------##-----##
## #How does the distribution of diamond prices vary with respect to different diamond shapes?
# Creating box plots for the distribution of diamond prices by shape

ggplot(diamonds, aes(x=shape, y=total_sales_price)) +
  geom_boxplot() +
  ggtitle("Distribution of Diamond Prices by Shape") +
  xlab("Diamond Shape") +
  ylab("Price ($)") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  scale_y_continuous(limits = c(0, 75000), expand = c(0, 0))
# 3082 values are dropped as they are do not fit the graph, but to maintain the
# sanctity of the plot this has to me maintained. 


##What is the relationship between diamond size and price? Is it a linear relationship?
# Creating  the scatter plot
ggplot(diamonds, aes(x=size, y=total_sales_price, color=color)) +
  geom_point() +
  ggtitle("Diamond Size vs. Price") +
  xlab("Diamond Size (carats)") +
  ylab("Price ($)") +
  theme(legend.position = "bottom") +
  scale_color_brewer(palette = "Set1")

# Fitting a linear regression model
model <- lm(total_sales_price ~ size, data = diamonds)
summary(model)

# Adding the regression line to the plot
ggplot(diamonds, aes(x=size, y=total_sales_price, color=color)) +
  geom_point() +
  ggtitle("Diamond Size vs. Price") +
  xlab("Diamond Size (carats)") +
  ylab("Price ($)") +
  ylim(0,75000) +
  xlim(0,6) +
  theme(legend.position = "bottom") +
  scale_color_brewer(palette = "Set1") +
  geom_smooth(method = "lm", se = FALSE)
# 11886 rows are omitted as they have missing values.

# Creating a histogram for the distribution of the cullet condition
ggplot(diamonds, aes(x=culet_condition)) +
  geom_bar(color='black', fill='lightblue', alpha=0.8) +
  labs(x='Culet Condition', y='Frequency', title='Distribution of Culet Conditions') +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5))

##--------##----- Enhancing Data Processing Capabilities --------##---------##
# Converting categorical variables into binary variables 

# Create a duplicate of the diamonds dataset
diamonds_dup <- diamonds

cols_to_dummy <- c("shape","color","cut","clarity","symmetry","polish","girdle_min",
                   "girdle_max","culet_size","culet_condition","fluor_color","fluor_intensity","lab")

diamonds <- dummy_cols(diamonds, select_columns = cols_to_dummy)

# Drop the 'date' column from the diamonds dataset
diamonds <- diamonds[, !colnames(diamonds) %in% c("date","shape","color","clarity","symmetry","polish","girdle_min","girdle_max","culet_size","culet_condition","fluor_color","fluor_intensity","lab","cut")]


##------##------##------ Running Regression Models ------##-------##-------##
## LINEAR REGRESSION


# Splitting the data into features and target variable
X <- diamonds %>% select(-diamond_id, -total_sales_price)
y <- diamonds$total_sales_price
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]


# Normalizing the input data
preprocessParams <- preProcess(X_train, method = c("center", "scale"))
X_train_norm <- predict(preprocessParams, X_train)
X_test_norm <- predict(preprocessParams, X_test)
y_train_norm <- scale(y_train)
y_test_norm <- scale(y_test)

# Fitting a linear regression model
linear <- lm(y_train_norm ~ ., data = X_train_norm)

# Making predictions on the test set
y_pred_norm <- predict(linear, X_test_norm)

# Calculating the MSE, MAE, and RMSE
mse <- mean((y_test_norm - y_pred_norm)^2)
mae <- mean(abs(y_test_norm - y_pred_norm))
rmse <- sqrt(mse)
r2 <- cor(y_test_norm, y_pred_norm)^2

# Printing the results
cat("MSE: ", mse, "\n")
cat("MAE: ", mae, "\n")
cat("RMSE: ", rmse, "\n")
cat("R2 Score: ", r2, "\n")



## LASSO REGRESSION

# Converting X_train_norm and X_test_norm to a matrix
X_train_norm_mat <- as.matrix(X_train_norm)
X_test_norm_mat <- as.matrix(X_test_norm)

# Fitting the Lasso model using matrix input
lasso <- glmnet(X_train_norm_mat, y_train_norm, alpha = 1, lambda = 0.01)

# Making predictions on the test set
y_pred_norm <- predict(lasso, X_test_norm_mat)

# Calculating the MSE, MAE, and RMSE
mse <- mean((y_test_norm - y_pred_norm)^2)
mae <- mean(abs(y_test_norm - y_pred_norm))
rmse <- sqrt(mse)
r2 <- cor(y_test_norm, y_pred_norm)^2

# Printing the results
cat("MSE: ", mse, "\n")
cat("MAE: ", mae, "\n")
cat("RMSE: ", rmse, "\n")
  cat("R2 Score: ", r2, "\n")


##RIDGE REGRESSION

# Fitting the Lasso model using matrix input
ridge <- glmnet(X_train_norm_mat, y_train_norm, alpha=0, lambda=0.001)

# Make predictions on the test set
y_pred_norm <- predict(ridge, X_test_norm_mat)

# Calculating the MSE, MAE, and RMSE
mse <- mean((y_test_norm - y_pred_norm)^2)
mae <- mean(abs(y_test_norm - y_pred_norm))
rmse <- sqrt(mse)
r2 <- cor(y_test_norm, y_pred_norm)^2

# Printing the results
cat("MSE: ", mse, "\n")
cat("MAE: ", mae, "\n")
cat("RMSE: ", rmse, "\n")
cat("R2 Score: ", r2, "\n")

