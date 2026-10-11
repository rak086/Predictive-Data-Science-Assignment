# Question 7: Predictive Analysis using R to diagnose Breast Cancer

# Simulating medical predictor features and diagnosis outcome
set.seed(42)
n <- 200

# Features: Clump Thickness, Cell Size, Cell Shape, Marginal Adhesion
clump_thickness <- sample(1:10, n, replace = TRUE)
cell_size <- sample(1:10, n, replace = TRUE)
cell_shape <- sample(1:10, n, replace = TRUE)
marginal_adhesion <- sample(1:10, n, replace = TRUE)

# Probability of malignancy based on predictors
log_odds <- -10 + 0.8 * clump_thickness + 0.7 * cell_size + 0.6 * cell_shape
prob <- 1 / (1 + exp(-log_odds))
diagnosis <- ifelse(prob > 0.5, "Malignant", "Benign")
diagnosis <- as.factor(diagnosis)

cancer_data <- data.frame(clump_thickness, cell_size, cell_shape, marginal_adhesion, diagnosis)

# Splitting train and test sets (70:30)
train_indices <- sample(1:n, 0.7 * n)
train_set <- cancer_data[train_indices, ]
test_set <- cancer_data[-train_indices, ]

# Logistic Regression Model for Diagnosis
model <- glm(diagnosis ~ ., data = train_set, family = binomial)
summary(model)

# Predictions on test data
pred_probs <- predict(model, newdata = test_set, type = "response")
predictions <- ifelse(pred_probs > 0.5, "Malignant", "Benign")

# Confusion Matrix
conf_matrix <- table(Predicted = predictions, Actual = test_set$diagnosis)
print("--- Confusion Matrix ---")
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Model Accuracy:", accuracy * 100, "%\n")
