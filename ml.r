library("arm")
train_data = relevant_data_final[1:1120,]
test_data = relevant_data_final[1121:1451,]

model<-lm(train_data$SalePrice~.,data = train_data)
summary(model)
prediction_linear_regression <- predict(model,newdata = test_data)
RMSE_linear_regression = sqrt((sum((test_data$SalePrice - prediction_linear_regression) ^ 2))/331)

model1 <- bayesglm(formula = train_data$SalePrice ~ ., family = gaussian, data = train_data, prior.mean = 0, prior.scale = Inf, prior.df = Inf)
summary(model1)
prediction_bayesian <- predict(model1,newdata = test_data)
RMSE_bayesian = sqrt((sum((test_data$SalePrice - prediction_bayesian) ^ 2))/331)

