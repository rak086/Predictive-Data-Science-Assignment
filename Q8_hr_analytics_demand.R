# Question 8: HR Analytics - Predict demand for hourly employees for next month in R

# Simulating historical monthly HR data
# Predictors: Production Volume, Seasonal Factor, Past Overtime Hours, Demand (hours needed)
set.seed(123)
months <- 1:24 # Past 24 months
production_volume <- round(rnorm(24, mean = 5000, sd = 500))
overtime_hours <- round(rnorm(24, mean = 300, sd = 40))

# Demand for hourly employees (dependent variable)
demand <- 20 + 0.015 * production_volume + 0.05 * overtime_hours + rnorm(24, 0, 3)

hr_data <- data.frame(Month = months, Production = production_volume, Overtime = overtime_hours, Demand = demand)

# Linear Regression Model for HR Demand
hr_model <- lm(Demand ~ Production + Overtime, data = hr_data)
summary(hr_model)

# Next month forecasted conditions
next_month <- data.frame(Production = 5300, Overtime = 320)

predicted_demand <- predict(hr_model, newdata = next_month)
cat("\n--- HR Analytics Hourly Employee Demand Prediction ---\n")
cat("Projected Employee Demand for Following Month:", round(predicted_demand, 2), "workers\n")
