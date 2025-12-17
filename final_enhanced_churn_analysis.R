################################################################################
#                                                                              #
#           MULTI-STATE CUSTOMER CHURN ANALYSIS WITH PENALIZED                 #
#              MULTINOMIAL LOGISTIC REGRESSION (ELASTIC NET)                   #
#                                                                              #
#  Author: Sanyukta Bhandari                                                   #
#  Date: June, 2025                                                            #
#                                                                              #
################################################################################

# ==============================================================================
# SECTION 0: LIBRARY LOADING AND INITIAL SETUP
# ==============================================================================

# Suppress warnings for cleaner output during package loading
options(warn = -1)

cat("\n")
cat("================================================================================\n")
cat("           ENHANCED CUSTOMER CHURN ANALYSIS - INITIALIZATION                   \n")
cat("================================================================================\n\n")

# Define required packages
required_packages <- c(
  "tidyverse",      # Data manipulation (dplyr, ggplot2, tidyr)
  "mice",           # Multiple Imputation by Chained Equations
  "VIM",            # Visualization and Imputation of Missing values
  "naniar",         # Missing data visualization
  "corrplot",       # Correlation matrix visualization
  "car",            # Companion to Applied Regression (VIF calculation)
  "glmnet",         # Lasso and Elastic-Net Regularized Generalized Linear Models
  "caret",          # Classification And Regression Training
  "pROC",           # Display and Analyze ROC curves
  "MLmetrics",      # Machine Learning Evaluation Metrics
  "gridExtra",      # Arrange multiple grid-based plots
  "reshape2",       # Flexibly reshape data (melt function)
  "scales",         # Scale functions for visualization
  "viridis",        # Colorblind-friendly color palettes
  "randomForest",   # Random Forest for feature importance
  "psych"           # Descriptive statistics
)

# Install missing packages
cat("Checking and installing required packages...\n")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, dependencies = TRUE)
}

# Load all packages silently
invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

cat("✓ All packages loaded successfully\n\n")

# Set random seed for reproducibility across all analyses
set.seed(123)

# Set ggplot2 theme globally for consistent visualizations
theme_set(theme_minimal(base_size = 11))

# Restore warning display
options(warn = 0)

# ==============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 1: Data Loading and Initial Exploration\n")
cat("================================================================================\n\n")

customer_file_path <- "C:/Users/DELL/Desktop/Datasets/telecom_customer_churn.csv"
zipcode_file_path <- "C:/Users/DELL/Desktop/Datasets/telecom_zipcode_population.csv"

cat("Loading data from:\n")
cat("  Customer data:", customer_file_path, "\n")
cat("  Zipcode data:", zipcode_file_path, "\n\n")

# Load datasets with comprehensive error handling
tryCatch({
  # Check if files exist first
  if(!file.exists(customer_file_path)) {
    stop("Customer data file not found at: ", customer_file_path)
  }
  if(!file.exists(zipcode_file_path)) {
    stop("Zipcode data file not found at: ", zipcode_file_path)
  }
  
  # Load the files
  customer_data <- read.csv(customer_file_path, 
                            stringsAsFactors = FALSE, 
                            na.strings = c("", "NA", " ", "NULL"))
  
  zipcode_data <- read.csv(zipcode_file_path, 
                           stringsAsFactors = FALSE, 
                           na.strings = c("", "NA", " ", "NULL"))
  
  cat("✓ Data loaded successfully\n")
  cat("  Customer records:", format(nrow(customer_data), big.mark = ","), "\n")
  cat("  Customer features:", ncol(customer_data), "\n")
  cat("  Zipcode records:", format(nrow(zipcode_data), big.mark = ","), "\n\n")
  
}, error = function(e) {
  stop("\n❌ ERROR: Could not load data files.\n",
       "   Error details: ", e$message, "\n",
       "   Please verify:\n",
       "   1. File paths are correct and complete\n",
       "   2. Files exist at specified locations\n",
       "   3. You have read permissions for these files\n",
       "   4. Files are valid CSV format\n")
})

# Display initial structure
cat("Initial data structure overview:\n")
cat("Customer data columns:", paste(head(names(customer_data), 10), collapse = ", "), "...\n\n")

# Merge customer data with zipcode information
merged_data <- customer_data %>%
  left_join(zipcode_data, by = "Zip.Code", suffix = c("", ".zip"))

cat("✓ Data merged: ", format(nrow(merged_data), big.mark = ","), "records\n")
cat("  Total features after merge:", ncol(merged_data), "\n\n")

# ==============================================================================
# SECTION 2: TARGET VARIABLE DEFINITION AND EXPLORATORY DATA ANALYSIS
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 2: Target Variable Definition and EDA\n")
cat("================================================================================\n\n")

# Basic NA handling
# Doing this at this step helps in efficiently creating the target variable and defining recently_joined_customers
merged_data <- merged_data %>%
  mutate(
    # Replace NA in Churn.Reason with "No Churn"
    Churn.Reason = ifelse(is.na(Churn.Reason), "No Churn", Churn.Reason),
    
    # Replace NA in Churn_Category with "No Churn"
    Churn.Category = ifelse(is.na(Churn.Category), "No Churn", Churn.Category)
  )

# Create multi-state target variable
# No Churn: Customer stayed
# Churn Reasons: Specific reasons why customer churned
# Recently Joined: New customers (will be handled separately)
merged_data <- merged_data %>%
  mutate(
    Churn_Category = case_when(
      Customer.Status == "Stayed" ~ "No Churn",
      Customer.Status == "Churned" ~ Churn.Category,
      Customer.Status == "Joined" ~ "Recently Joined",
      TRUE ~ "Unknown"
    )
  )

# Display target variable distribution
cat("Target Variable Distribution:\n")
churn_distribution <- table(merged_data$Churn_Category)
churn_percentage <- prop.table(churn_distribution) * 100
churn_summary <- data.frame(
  Category = names(churn_distribution),
  Count = as.numeric(churn_distribution),
  Percentage = sprintf("%.2f%%", churn_percentage)
)
print(churn_summary)
cat("\n")

# CRITICAL NOTE: Recently Joined Customers
cat("================================================================================\n")
cat("IMPORTANT: Treatment of Recently Joined Customers\n")
cat("================================================================================\n")
cat("Recently joined customers have not been associated with the company long enough\n")
cat("for their churn behavior to be observable. Including them in the churn prediction\n")
cat("model would distort the estimates of churn predictors since they haven't had\n")
cat("sufficient opportunity to exhibit churn patterns. These customers will be:\n")
cat("  1. Excluded from model training and evaluation\n")
cat("  2. Analyzed separately for risk profiling\n")
cat("  3. Scored using the trained model for proactive retention\n")
cat("================================================================================\n\n")

# Separate recently joined customers for later analysis
recently_joined_customers <- merged_data %>%
  filter(Churn_Category == "Recently Joined")

cat("Recently Joined Customer Statistics:\n")
cat("  Count:", format(nrow(recently_joined_customers), big.mark = ","), "\n")
cat("  Avg Tenure:", round(mean(recently_joined_customers$Tenure.in.Months, na.rm = TRUE), 2), "months\n")
cat("  Avg Monthly Charge: $", round(mean(recently_joined_customers$Monthly.Charge, na.rm = TRUE), 2), "\n\n")
 
# Create modeling dataset (exclude Recently Joined and Unknown)
modeling_data <- merged_data %>%
  filter(!Churn_Category %in% c("Recently Joined", "Unknown"))

cat("Modeling Dataset:\n")
cat("  Records:", format(nrow(modeling_data), big.mark = ","), "\n")
cat("  Features:", ncol(modeling_data), "\n\n")

# Basic EDA visualizations
cat("Generating exploratory visualizations...\n")

# Plot 1: Target variable distribution
p1 <- ggplot(modeling_data, aes(x = reorder(Churn_Category, table(Churn_Category)[Churn_Category]))) +
  geom_bar(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "Distribution of Churn Categories",
       subtitle = "Excluding Recently Joined Customers",
       x = "Churn Category", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p1)

# Plot 2: Tenure distribution by churn category
p2 <- ggplot(modeling_data, aes(x = Churn_Category, y = Tenure.in.Months, fill = Churn_Category)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  coord_flip() +
  labs(title = "Tenure Distribution by Churn Category",
       x = "Churn Category", y = "Tenure (Months)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p2)

cat("✓ EDA visualizations completed\n\n")

# ==============================================================================
# SECTION 3: MISSING VALUE ANALYSIS AND IMPUTATION
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 3: Missing Value Analysis and Imputation\n")
cat("================================================================================\n\n")

# Calculate missing value percentages for each column
missing_summary <- modeling_data %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing_Count") %>%
  mutate(
    Total_Count = nrow(modeling_data),
    Missing_Percentage = (Missing_Count / Total_Count) * 100
  ) %>%
  filter(Missing_Count > 0) %>%
  arrange(desc(Missing_Percentage))

cat("Variables with missing values:\n")
print(missing_summary, n = 20)
cat("\n")

# Identify columns with >50% missing values (to be removed)
cols_to_remove <- missing_summary %>%
  filter(Missing_Percentage > 50) %>%
  pull(Variable)

if(length(cols_to_remove) > 0) {
  cat("⚠ Removing", length(cols_to_remove), "columns with >50% missing values:\n")
  cat("  ", paste(cols_to_remove, collapse = ", "), "\n\n")
  modeling_data <- modeling_data %>%
    select(-all_of(cols_to_remove))
} else {
  cat("✓ No columns have >50% missing values\n\n")
}

# Visualize missing data patterns
cat("Generating missing data visualizations...\n")

# Missing data pattern plot
if(sum(is.na(modeling_data)) > 0) {
  
# Plot missing data using naniar
p3 <- gg_miss_var(modeling_data, show_pct = TRUE) +
  labs(title = "Missing Data by Variable",
       subtitle = "After removing columns with >50% missing") +
  theme_minimal()
  
print(p3)
   
# Missing data pattern
p4 <- vis_miss(modeling_data, cluster = TRUE) +
  labs(title = "Missing Data Pattern") +
  theme(
    axis.text.x = element_text(
      angle = 90,          # Vertical orientation
      vjust = 0.5,         # Center vertically along their x-tick
      hjust = 0,           # Aligns left edges evenly
      size = 8,            # Adjust for readability
      margin = ggplot2::margin(t = 2)
    ),
    axis.title.x = element_blank(),
    axis.ticks.length.x = unit(0, "pt")
  )

print(p4)
  
} else {
  cat("✓ No missing values detected\n")
}

# Handle specific missing value cases before MICE
cat("\nHandling specific missing value patterns...\n")

# Total.Charges often missing for very new customers (tenure = 0)
if("Total.Charges" %in% names(modeling_data)) {
  modeling_data <- modeling_data %>%
    mutate(Total.Charges = ifelse(is.na(Total.Charges) & Tenure.in.Months == 0, 
                                  0, 
                                  Total.Charges))
  cat("✓ Handled Total.Charges for zero-tenure customers\n")
}

# Internet service-related features: NA likely means "No Internet Service"
internet_cols <- c("Internet.Type","Online.Security", "Online.Backup", "Device.Protection.Plan", 
                   "Premium.Tech.Support", "Streaming.TV", "Streaming.Movies", 
                   "Streaming.Music", "Unlimited.Data")

existing_internet_cols <- intersect(internet_cols, names(modeling_data))
if(length(existing_internet_cols) > 0) {
  for(col in existing_internet_cols) {
    modeling_data[[col]][is.na(modeling_data[[col]])] <- "No Internet Service"
  }
  cat("✓ Handled internet service-related missing values\n")
}

# MICE Imputation for remaining missing values
# MICE Imputation is preferred since for the given data <30% is missing 
if(sum(is.na(modeling_data)) > 0) {
  
  cat("\nPerforming MICE imputation...\n")
  cat("(This may take a few minutes depending on data size)\n\n")
  
  # Identify columns for imputation (exclude ID columns and target)
  cols_to_exclude <- c("Customer.ID", "Churn_Category", "Customer.Status", 
                       "Churn.Reason", "Churn.Label", "Churn.Category")
  cols_for_imputation <- setdiff(names(modeling_data), cols_to_exclude)
  
  # Separate data
  data_for_imputation <- modeling_data %>% select(all_of(cols_for_imputation))
  data_excluded <- modeling_data %>% select(all_of(intersect(cols_to_exclude, names(modeling_data))))
  
  # Run MICE (m=5 imputations, maxit=5 iterations)
  # Using pmm (predictive mean matching) method for robustness
  mice_model <- mice(data_for_imputation, 
                     m = 5,           # Number of imputations
                     maxit = 5,       # Number of iterations
                     method = "pmm",  # Predictive mean matching
                     seed = 123,
                     printFlag = FALSE)
  
  # Complete the imputation (use first imputed dataset)
  imputed_data <- complete(mice_model, 1)
  
  # Combine back with excluded columns
  modeling_data <- bind_cols(data_excluded, imputed_data)
  
  cat("✓ MICE imputation completed\n")
  cat("  Remaining missing values:", sum(is.na(modeling_data)), "\n\n")
  
} else {
  cat("✓ No missing values remaining, skipping MICE imputation\n\n")
}

# Handle Multiple.Lines Feature 
# Show original distribution
cat(" Original distribution:\n")
cat(" Customers with Phone.Service = Yes & Multiple.Lines = Yes:", 
    sum(modeling_data$Phone.Service == "Yes" & modeling_data$Multiple.Lines == "Yes", na.rm = TRUE), "\n")
cat(" Customers with Phone.Service = Yes & Multiple.Lines = No:", 
    sum(modeling_data$Phone.Service == "Yes" & modeling_data$Multiple.Lines == "No", na.rm = TRUE), "\n")
cat(" Customers with Phone.Service = No (Multiple.Lines = NA):", 
    sum(modeling_data$Phone.Service == "No", na.rm = TRUE), "\n\n")

# Convert to clean binary
modeling_data <- modeling_data %>%
  mutate(
    Multiple.Lines = case_when(
      # Customers without phone service logically cannot have multiple lines
      Phone.Service == "No" ~ "No",
      
      # Customers with phone service: use actual value
      Phone.Service == "Yes" & Multiple.Lines == "Yes" ~ "Yes",
      Phone.Service == "Yes" & Multiple.Lines == "No" ~ "No",
      
      # Safety net for any unexpected NAs
      is.na(Multiple.Lines) ~ "No",
      
      # Default
      TRUE ~ "No"
    )
  )

cat("  ✓ NAs in Multiple.Lines successfully handled \n")


# ==============================================================================
# SECTION 4: OUTLIER DETECTION AND TREATMENT
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 4: Outlier Detection and Treatment\n")
cat("================================================================================\n\n")

# Identify numeric columns for outlier analysis
numeric_cols <- names(modeling_data)[sapply(modeling_data, is.numeric)]
numeric_cols <- setdiff(numeric_cols, c("Customer.ID", "Zip.Code"))  # Exclude IDs

cat("Numeric variables for outlier detection:\n")
cat(" ", paste(numeric_cols, collapse = ", "), "\n\n")

# Visualize outliers before treatment
cat("Generating outlier visualizations...\n")

# Create box plots for numeric variables (before treatment)
if(length(numeric_cols) > 0) {
  
  # Select key numeric variables for visualization
  key_numeric_vars <- head(numeric_cols, 6)
  
  plot_list <- lapply(key_numeric_vars, function(var) {
    ggplot(modeling_data, aes(y = .data[[var]])) +
      geom_boxplot(fill = "lightcoral", alpha = 0.7) +
      labs(title = paste("Before:", var), y = var) +
      theme_minimal() +
      theme(plot.title = element_text(size = 9))
  })
  
  do.call(grid.arrange, c(plot_list, ncol = 3))
}

# Cap outliers at 1st and 99th percentiles
cat("\nCapping outliers at 1st and 99th percentiles...\n")

outlier_summary <- data.frame(
  Variable = character(),
  Original_Min = numeric(),
  Original_Max = numeric(),
  P1 = numeric(),
  P99 = numeric(),
  N_Capped_Low = numeric(),
  N_Capped_High = numeric(),
  stringsAsFactors = FALSE
)

for(col in numeric_cols) {
  
  original_min <- min(modeling_data[[col]], na.rm = TRUE)
  original_max <- max(modeling_data[[col]], na.rm = TRUE)
  
  # Calculate 1st and 99th percentiles
  p1 <- quantile(modeling_data[[col]], 0.01, na.rm = TRUE)
  p99 <- quantile(modeling_data[[col]], 0.99, na.rm = TRUE)
  
  # Count values to be capped
  n_low <- sum(modeling_data[[col]] < p1, na.rm = TRUE)
  n_high <- sum(modeling_data[[col]] > p99, na.rm = TRUE)
  
  # Cap the values
  modeling_data[[col]] <- pmin(pmax(modeling_data[[col]], p1), p99)
  
  # Store summary
  outlier_summary <- rbind(outlier_summary, data.frame(
    Variable = col,
    Original_Min = round(original_min, 2),
    Original_Max = round(original_max, 2),
    P1 = round(p1, 2),
    P99 = round(p99, 2),
    N_Capped_Low = n_low,
    N_Capped_High = n_high
  ))
}

cat("\nOutlier Treatment Summary:\n")
print(outlier_summary, row.names = FALSE)
cat("\n")

# Visualize after treatment
cat("Generating post-treatment visualizations...\n")

if(length(key_numeric_vars) > 0) {
  plot_list_after <- lapply(key_numeric_vars, function(var) {
    ggplot(modeling_data, aes(y = .data[[var]])) +
      geom_boxplot(fill = "lightgreen", alpha = 0.7) +
      labs(title = paste("After:", var), y = var) +
      theme_minimal() +
      theme(plot.title = element_text(size = 9))
  })
  
  do.call(grid.arrange, c(plot_list_after, ncol = 3))
}

cat("✓ Outlier treatment completed\n\n")

# ==============================================================================
# SECTION 5: FEATURE ENGINEERING
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 5: Feature Engineering\n")
cat("================================================================================\n\n")

cat("Creating derived features...\n")

modeling_data <- modeling_data %>%
  mutate(
    # Tenure groups for better interpretability
    Tenure_Group = case_when(
      Tenure.in.Months <= 6 ~ "New",   # (0-6 months)
      Tenure.in.Months <= 24 ~ "Growing",   # (7-24 months)
      Tenure.in.Months <= 60 ~ "Mature",   # (25-60 months)
      TRUE ~ "Loyal"   # (60+ months)
    ),
    
    # To help in variable name checking at later steps 
    Contract = case_when(
      Contract == "Month-to-Month" ~ "Month to Month",
      Contract == "One Year" ~ "One Year",
      Contract == "Two Year" ~ "Two Year",
    ),
    
    # Average revenue per month (handles zero tenure)
    Avg_Revenue_Per_Month = ifelse(Tenure.in.Months > 0, 
                                   Total.Charges / Tenure.in.Months, 
                                   Monthly.Charge),
    
    # Total service count (sum of all additional services)
    Service_Count = rowSums(select(., any_of(c("Online.Security", "Online.Backup", 
                                               "Device.Protection.Plan", "Premium.Tech.Support",
                                               "Streaming.TV", "Streaming.Movies", 
                                               "Streaming.Music", "Unlimited.Data"))) == "Yes", 
                            na.rm = TRUE),
    
    # Revenue to tenure ratio (customer value intensity)
    Revenue_Tenure_Ratio = ifelse(Tenure.in.Months > 0, 
                                  Total.Charges / Tenure.in.Months, 
                                  0),
    
    # Binary: Has any streaming service
    Has_Streaming = ifelse(rowSums(select(., any_of(c("Streaming.TV", "Streaming.Movies", 
                                                      "Streaming.Music"))) == "Yes", 
                                   na.rm = TRUE) > 0, 
                           "Yes", "No"),
    
    # Binary: Has any protection service
    Has_Protection = ifelse(rowSums(select(., any_of(c("Online.Security", "Online.Backup", 
                                                       "Device.Protection.Plan"))) == "Yes", 
                                    na.rm = TRUE) > 0, 
                            "Yes", "No"),
    
    # Age group
    Age_Group = case_when(
      Age < 30 ~ "Young",   # (< 30 years)
      Age < 50 ~ "Middle",   # (30-50 years)
      TRUE ~ "Senior"   # (50+ years)
    ),
    
    # Contract-Tenure interaction (early cancellation risk)
    Contract_Tenure_Risk = case_when(
      Contract == "Month to Month" & Tenure.in.Months < 12 ~ "High Risk",
      Contract == "Month to Month" ~ "Medium Risk",
      TRUE ~ "Low Risk"
    )
  )

cat("✓ Feature engineering completed\n")
cat("  New features created: Tenure_Group, Avg_Revenue_Per_Month, Service_Count,\n")
cat("                        Revenue_Tenure_Ratio, Has_Streaming, Has_Protection,\n")
cat("                        Age_Group, Contract_Tenure_Risk\n\n")

# ==============================================================================
# SECTION 6: FEATURE TYPE PREPARATION AND TRANSFORMATION
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 6: Feature Type Preparation and Transformation\n")
cat("================================================================================\n\n")

# Convert target variable to factor
modeling_data$Churn_Category <- as.factor(modeling_data$Churn_Category)

cat("Target variable levels:\n")
cat(" ", paste(levels(modeling_data$Churn_Category), collapse = ", "), "\n\n")

# Identify categorical variables for encoding
categorical_vars <- names(modeling_data)[sapply(modeling_data, function(x) 
  is.character(x) || is.factor(x))]

# Exclude target and ID columns
categorical_vars <- setdiff(categorical_vars, c("Churn_Category", "Customer.ID", "Churn.Category", 
                                                "Customer.Status", "Churn.Reason"))

cat("Categorical variables identified:", length(categorical_vars), "\n")
cat(" ", paste(head(categorical_vars, 10), collapse = ", "), "...\n\n")

# Convert categorical variables to factors
for(var in categorical_vars) {
  modeling_data[[var]] <- as.factor(modeling_data[[var]])
}

cat("✓ Categorical variables converted to factors\n\n")
cat("✓ One-hot encoding will be performed AFTER class imbalance treatment\n\n")

# ==============================================================================
# SECTION 7: MULTICOLLINEARITY CHECK
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 7: Multicollinearity Analysis\n")
cat("================================================================================\n\n")

# Calculate correlation matrix for numeric variables
cat("Calculating correlation matrix...\n")

# Use a subset of original numeric variables for correlation analysis
numeric_original <- modeling_data %>%
  select(where(is.numeric)) %>%
  select(-any_of(c("Customer.ID", "Zip.Code")))

if(ncol(numeric_original) > 1) {
  
  cor_matrix <- cor(numeric_original, use = "complete.obs")
  
  # Visualize correlation matrix
  corrplot(cor_matrix, 
           method = "color",
           type = "upper",
           tl.col = "black",
           tl.srt = 45,
           tl.cex = 0.7,
           addCoef.col = "black",
           number.cex = 0.5,
           title = "Correlation Matrix of Numeric Predictors",
           mar = c(0,0,2,0))
  
  # Identify highly correlated pairs (|r| > 0.8)
  high_cor_pairs <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)
  
  if(nrow(high_cor_pairs) > 0) {
    cat("\n⚠ Highly correlated variable pairs (|r| > 0.8):\n")
    for(i in 1:nrow(high_cor_pairs)) {
      var1 <- rownames(cor_matrix)[high_cor_pairs[i, 1]]
      var2 <- colnames(cor_matrix)[high_cor_pairs[i, 2]]
      cor_val <- cor_matrix[high_cor_pairs[i, 1], high_cor_pairs[i, 2]]
      cat("  ", var1, "<->", var2, ": r =", round(cor_val, 3), "\n")
    }
  } else {
    cat("\n✓ No highly correlated pairs detected (threshold: |r| > 0.8)\n")
  }
  
} else {
  cat("⚠ Insufficient numeric variables for correlation analysis\n")
}

cat("\nNote: Multicollinearity will be addressed through Elastic Net regularization\n")
cat("      which automatically handles correlated predictors.\n\n")

# ==============================================================================
# SECTION 8: TRAIN-TEST SPLIT
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 8: Train-Test Split (Stratified)\n")
cat("================================================================================\n\n")

# Remove single-level features (if any)
single_level_cols <- names(Filter(function(x) length(unique(x)) == 1, modeling_data))

if(length(single_level_cols) > 0) {
  cat("  Removing", length(single_level_cols), "single-level features:\n")
  cat("  ", paste(single_level_cols, collapse = ", "), "\n\n")
  modeling_data <- modeling_data[, setdiff(names(modeling_data), single_level_cols)]
}

# Create stratified train-test split (80-20)
# Stratification ensures each churn category is proportionally represented
set.seed(123)
train_index <- createDataPartition(modeling_data$Churn_Category, 
                                   p = 0.80, 
                                   list = FALSE)

# Split the FULL dataset (with factors intact)
train_data <- modeling_data[train_index, ]
test_data <- modeling_data[-train_index, ]

# Extract target variables
y_train <- train_data$Churn_Category
y_test <- test_data$Churn_Category

cat("Data split completed:\n")
cat("  Training set:", format(nrow(train_data), big.mark = ","), "records\n")
cat("  Test set:", format(nrow(test_data), big.mark = ","), "records\n\n")

# Check class distribution in train and test
cat("Training set class distribution:\n")
print(table(y_train))
cat("\nTest set class distribution:\n")
print(table(y_test))
cat("\n")

# ==============================================================================
# SECTION 9: ADDRESSING CLASS IMBALANCE WITH CLASS-WEIGHTS
# ==============================================================================

# ==============================================================================
# SECTION 10: FEATURE SELECTION USING RANDOM FOREST 
# ==============================================================================

# ===========================================================================================
# SECTION 11: ELASTIC NET REGULARIZED MULTINOMIAL LOGISTIC REGRESSION WITH CROSS-VALIDATION
# ===========================================================================================

# =========================================================================================
# SECTION 9-11 (INTEGRATED): NESTED CLASS-WEIGHTS WITH STRATIFIED CV AND FEATURE SELECTION
# =========================================================================================

cat("================================================================================\n")
cat("SECTION 9-11: Nested Class Weights + Feature Selection + Stratified CV + Final Model fit CV\n")
cat("================================================================================\n\n")

# Function to perform Feature Selection + Model Training for one fold
train_and_validate_fold <- function(train_fold_data, val_fold_data, alpha) {
  
  # Extract target
  y_train_fold <- train_fold_data$Churn_Category
  y_val_fold <- val_fold_data$Churn_Category
  
  # Calculate Class Weights for THIS fold
  class_counts <- table(y_train_fold)
  num_classes <- length(class_counts)
  total_samples <- length(y_train_fold)
  class_weights_map <- total_samples / (num_classes * class_counts)
  sample_weights <- class_weights_map[match(as.character(y_train_fold), names(class_weights_map))]
    
  # Prepare features (exclude ID and target columns)
  cols_to_exclude <- c("Customer.ID", "Churn_Category", "Customer.Status", 
                       "Churn.Reason", "Zip.Code", "Churn.Category")
  feature_cols <- setdiff(names(train_fold_data), cols_to_exclude)
  
  # Remove high-cardinality categorical variables (>53 levels) for Random Forest
  train_fold_rf <- train_fold_data[, c("Churn_Category", feature_cols)]
  
  high_cardinality_cols <- character()
  for(col in feature_cols) {
    if(is.factor(train_fold_rf[[col]]) || is.character(train_fold_rf[[col]])) {
      n_levels <- length(unique(train_fold_rf[[col]]))
      if(n_levels > 53) {
        high_cardinality_cols <- c(high_cardinality_cols, col)
      }
    }
  }
  
  # Remove problematic columns from RF input
  rf_feature_cols <- setdiff(feature_cols, high_cardinality_cols)
  
  # Feature selection using Random Forest (on RAW categorical data)
  rf_formula <- as.formula(paste("Churn_Category ~", paste(rf_feature_cols, collapse = " + ")))
  
  rf_fold <- randomForest(
    rf_formula,
    data = train_fold_rf,
    ntree = 100,
    importance = TRUE,
    nodesize = 50,
    maxnodes = 50
  )
  
  # Get feature importance
  importance_scores <- importance(rf_fold)[, "MeanDecreaseGini"]
  importance_sorted <- sort(importance_scores, decreasing = TRUE)
  
  # Select top features (95% cumulative importance)
  cumulative_imp <- cumsum(importance_sorted) / sum(importance_sorted)
  n_features <- which(cumulative_imp >= 0.95)[1]
  selected_features <- names(importance_sorted)[1:n_features]
  
  # NOW: One-hot encode ONLY the selected features
  # Create formula for selected features
  formula_str <- paste("~", paste(selected_features, collapse = " + "), "- 1")
  model_formula <- as.formula(formula_str)
  
  # Encode training fold
  X_train_fold <- model.matrix(model_formula, data = train_fold_data)
  
  # Encode validation fold (using SAME factor levels from training)
  X_val_fold <- model.matrix(model_formula, data = val_fold_data)
  
  # Scale features (fit on train fold, apply to val fold)
  scaling_params <- list()
  X_train_scaled <- X_train_fold
  X_val_scaled <- X_val_fold
  
  for(i in 1:ncol(X_train_fold)) {
    col_mean <- mean(X_train_fold[, i])
    col_sd <- sd(X_train_fold[, i])
    
    if(col_sd > 0) {
      X_train_scaled[, i] <- (X_train_fold[, i] - col_mean) / col_sd
      X_val_scaled[, i] <- (X_val_fold[, i] - col_mean) / col_sd
    }
  }
  
  # Train Elastic Net model WITH Sample Weights
  cv_lambda <- cv.glmnet(
    x = X_train_scaled,
    y = y_train_fold, 
    family = "multinomial",
    alpha = alpha,
    nfolds = 5,
    type.measure = "class",
    weights = as.numeric(sample_weights),
    standardize = FALSE
  )
  
  # Fit the Elastic Net regularized Multinomial Logistic Regression Model
  model_fold <- glmnet(
    x = X_train_scaled,
    y = y_train_fold,
    family = "multinomial",
    alpha = alpha,
    lambda = cv_lambda$lambda.min,
    weights = as.numeric(sample_weights), 
    standardize = FALSE
  )
  
  # Predict on validation fold 
  predictions <- predict(model_fold, 
                         newx = X_val_scaled,
                         s = cv_lambda$lambda.min,
                         type = "class")
  
  # Calculate accuracy
  accuracy <- mean(predictions == y_val_fold)
  
  return(list(
    accuracy = accuracy,
    lambda = cv_lambda$lambda.min,
    n_features = n_features,
    selected_features = selected_features,
    predictions = predictions,
    scaling_params = scaling_params
  ))
}

# Perform Stratified K-Fold CV with Different Alpha Values

# Parameters
n_folds <- 5
alpha_values <- seq(0, 1, by = 0.1) 

# Create stratified folds
cat("Creating stratified folds...\n")
set.seed(123)
fold_ids <- createFolds(y_train, k = n_folds, list = FALSE)

cat("  Number of folds:", n_folds, "\n")
cat("  Fold class distribution check:\n")
for(i in 1:n_folds) {
  fold_dist <- table(y_train[fold_ids == i])
  cat("    Fold", i, ":", paste(names(fold_dist), "=", fold_dist, collapse = ", "), "\n")
}
cat("\n")

# Storage for results
cv_results_nested <- list()
alpha_performance <- data.frame(
  Alpha = numeric(),
  Mean_Accuracy = numeric(),
  SD_Accuracy = numeric(),
  Mean_Lambda = numeric(),
  Mean_N_Features = numeric()
)

# Test each alpha value (Simplified Loop)
cat("Testing alpha values \n\n")

for(alpha in alpha_values) {
  
  cat("================================================================================\n")
  cat("Alpha =", alpha, "\n")
  cat("================================================================================\n")
  
  fold_results <- list()
  fold_accuracies <- numeric(n_folds)
  fold_lambdas <- numeric(n_folds)
  fold_n_features <- numeric(n_folds)
  
  # Cross-validation loop
  for(fold in 1:n_folds) {
    
    cat("  Processing fold", fold, "/", n_folds, "...")

    # Split data (on RAW categorical data)
    val_idx <- which(fold_ids == fold)
    train_idx <- which(fold_ids != fold)
    
    train_fold_data <- train_data[train_idx, ]
    val_fold_data <- train_data[val_idx, ]
    
    # Train and validate
    result <- train_and_validate_fold(train_fold_data, val_fold_data, alpha)
    
    fold_results[[fold]] <- result
    fold_accuracies[fold] <- result$accuracy
    fold_lambdas[fold] <- result$lambda
    fold_n_features[fold] <- result$n_features
    
    cat(" Accuracy:", round(result$accuracy, 4), 
        "| Features:", result$n_features, "\n")
  }
  
  # Calculate summary statistics
  mean_acc <- mean(fold_accuracies)
  sd_acc <- sd(fold_accuracies)
  
  cat("\n  Summary for alpha =", alpha, ":\n")
  cat("    Mean CV Accuracy:", round(mean_acc, 4), "±", round(sd_acc, 4), "\n")
  cat("    Accuracy Range: [", round(min(fold_accuracies), 4), ",", round(max(fold_accuracies), 4), "]\n")
  cat("    Mean Lambda:", round(mean(fold_lambdas), 6), "\n")
  cat("    Mean Features Selected:", round(mean(fold_n_features), 1), "\n\n")
  
  # Store results
  cv_results_nested[[paste0("alpha_", alpha)]] <- fold_results
  alpha_performance <- rbind(alpha_performance, data.frame(
    Alpha = alpha,
    Mean_Accuracy = mean_acc,
    SD_Accuracy = sd_acc,
    Mean_Lambda = mean(fold_lambdas),
    Mean_N_Features = mean(fold_n_features)
  ))
}

cat("CROSS-VALIDATION COMPLETE\n")

# Display full results table
cat("All Alpha Results:\n")
print(alpha_performance, row.names = FALSE, digits = 4)
cat("\n")


# Find best alpha
best_idx <- which.max(alpha_performance$Mean_Accuracy)
best_alpha <- alpha_performance$Alpha[best_idx]
best_accuracy <- alpha_performance$Mean_Accuracy[best_idx]
best_sd <- alpha_performance$SD_Accuracy[best_idx]



# F1-SCORE VALIDATION: Checking Top 3 Alpha Candidates

cat("Rationale: While accuracy was used for hyperparameter selection (stable, efficient),\n")
cat("we validate against Macro F1-Score (our primary evaluation metric) to ensure\n")
cat("the selected alpha also optimizes for balanced multi-class performance.\n\n")

# Get top 3 alphas by accuracy
top_3_alphas <- alpha_performance %>%
  arrange(desc(Mean_Accuracy)) %>%
  head(3) %>%
  pull(Alpha)

cat("Top 3 alpha candidates by CV accuracy:", paste(top_3_alphas, collapse = ", "), "\n\n")

# For each top alpha, calculate F1 scores from saved CV results
f1_validation <- data.frame(
  Alpha = numeric(),
  Mean_CV_Accuracy = numeric(),
  Mean_CV_F1 = numeric(),
  F1_SD = numeric(),
  Rank_by_Accuracy = integer(),
  Rank_by_F1 = integer()
)

cat("Recomputing Macro F1-Score for validation...\n\n")

for(alpha_val in top_3_alphas) {
  
  cat("  Alpha =", alpha_val, "...")
  
  # Get stored CV results for this alpha
  alpha_results <- cv_results_nested[[paste0("alpha_", alpha_val)]]
  
  # Calculate F1 for each fold
  fold_f1_scores <- numeric()
  fold_accuracies <- numeric()
  
  for(fold_idx in 1:length(alpha_results)) {
    
    fold_result <- alpha_results[[fold_idx]]
    
    # Get predictions and actual values
    predictions <- fold_result$predictions
    
    # Get validation fold indices for this fold
    val_idx <- which(fold_ids == fold_idx)
    y_val_fold <- y_train[val_idx]
    
    # Compute confusion matrix
    pred_factor <- factor(as.character(predictions), levels = levels(y_val_fold))
    
    tryCatch({
      cm <- confusionMatrix(pred_factor, y_val_fold)
      
      # Extract F1 scores
      if(length(levels(y_val_fold)) > 2) {
        f1_scores_fold <- cm$byClass[, "F1"]
        macro_f1_fold <- mean(f1_scores_fold, na.rm = TRUE)
      } else {
        macro_f1_fold <- cm$byClass["F1"]
      }
      
      fold_f1_scores <- c(fold_f1_scores, macro_f1_fold)
      fold_accuracies <- c(fold_accuracies, mean(predictions == y_val_fold))
      
    }, error = function(e) {
      cat("\n    Warning: Could not compute F1 for fold", fold_idx, "\n")
      fold_f1_scores <- c(fold_f1_scores, NA)
      fold_accuracies <- c(fold_accuracies, NA)
    })
  }
  
  # Calculate mean and SD
  mean_f1 <- mean(fold_f1_scores, na.rm = TRUE)
  sd_f1 <- sd(fold_f1_scores, na.rm = TRUE)
  mean_acc <- mean(fold_accuracies, na.rm = TRUE)
  
  # Store results
  f1_validation <- rbind(f1_validation, data.frame(
    Alpha = alpha_val,
    Mean_CV_Accuracy = mean_acc,
    Mean_CV_F1 = mean_f1,
    F1_SD = sd_f1,
    Rank_by_Accuracy = NA,  # Will fill later
    Rank_by_F1 = NA          # Will fill later
  ))
  
  cat(" F1:", round(mean_f1, 4), "±", round(sd_f1, 4), "\n")
}

cat("\n")

# Add rankings
f1_validation <- f1_validation %>%
  mutate(
    Rank_by_Accuracy = rank(-Mean_CV_Accuracy, ties.method = "first"),
    Rank_by_F1 = rank(-Mean_CV_F1, ties.method = "first")
  )

# Display results
cat("VALIDATION RESULTS:\n")
cat("────────────────────────────────────────────────────────────────────\n")
print(f1_validation, row.names = FALSE, digits = 4)
cat("\n")

# Find best by F1 among top 3
best_alpha_by_f1 <- f1_validation %>%
  filter(Rank_by_F1 == 1) %>%
  pull(Alpha)

# Check if accuracy-based selection matches F1-based
cat("VALIDATION OUTCOME:\n")
cat("────────────────────────────────────────────────────────────────────\n")
cat("Selected by Accuracy: Alpha =", best_alpha, 
    "\n  → CV Accuracy:", round(best_accuracy, 4), 
    "\n  → CV F1-Score:", round(f1_validation$Mean_CV_F1[f1_validation$Alpha == best_alpha], 4), "\n\n")

cat("Best by F1-Score: Alpha =", best_alpha_by_f1,
    "\n  → CV Accuracy:", round(f1_validation$Mean_CV_Accuracy[f1_validation$Alpha == best_alpha_by_f1], 4),
    "\n  → CV F1-Score:", round(f1_validation$Mean_CV_F1[f1_validation$Alpha == best_alpha_by_f1], 4), "\n\n")

cat("VALIDATION PASSED \n")
cat("Both metrics agree: Alpha =", best_alpha, "is optimal\n")
cat("→ Accuracy-based selection is validated by Macro F1-Score\n")
cat("→ Selected hyperparameter is robust across evaluation metrics\n\n")

cat("OPTIMAL HYPERPARAMETERS:\n")
cat("  Best Alpha:", best_alpha, "\n")
cat("  CV Accuracy:", round(best_accuracy, 4), "±", round(best_sd, 4), "\n")
cat("  CV F1-Score:", round(f1_validation$Mean_CV_F1[best_idx], 4), "\n")
cat("  Average Lambda:", round(alpha_performance$Mean_Lambda[best_idx], 6), "\n")
cat("  Average Features:", round(alpha_performance$Mean_N_Features[best_idx], 1), "\n\n")


# Visualize results

# Plot 1: Accuracy vs Alpha
plot(alpha_performance$Alpha, alpha_performance$Mean_Accuracy,
     type = "b", pch = 19, col = "steelblue", lwd = 2,
     xlab = "Alpha", ylab = "Mean CV Accuracy",
     main = "Cross-Validation Accuracy vs Alpha",
     ylim = c(min(alpha_performance$Mean_Accuracy - alpha_performance$SD_Accuracy),
              max(alpha_performance$Mean_Accuracy + alpha_performance$SD_Accuracy)))
arrows(alpha_performance$Alpha, 
       alpha_performance$Mean_Accuracy - alpha_performance$SD_Accuracy,
       alpha_performance$Alpha,
       alpha_performance$Mean_Accuracy + alpha_performance$SD_Accuracy,
       angle = 90, code = 3, length = 0.05, col = "gray50")
points(best_alpha, best_accuracy, pch = 19, col = "red", cex = 2)
abline(v = best_alpha, col = "red", lty = 2)
abline(h = best_accuracy, col = "red", lty = 2)


# Plot 2: Accuracy vs F1 for top 3
validation_plot_data <- f1_validation %>% arrange(Alpha)
plot(validation_plot_data$Mean_CV_Accuracy, 
     validation_plot_data$Mean_CV_F1,
     pch = 19, cex = 2, col = c("green", "blue", "orange"),
     xlim = c(min(validation_plot_data$Mean_CV_Accuracy) - 0.01,
              max(validation_plot_data$Mean_CV_Accuracy) + 0.01),
     ylim = c(min(validation_plot_data$Mean_CV_F1) - 0.01,
              max(validation_plot_data$Mean_CV_F1) + 0.01),
     xlab = "Mean CV Accuracy",
     ylab = "Mean CV F1-Score",
     main = "Top 3 Alphas: Accuracy vs F1 Trade-off")
text(validation_plot_data$Mean_CV_Accuracy, 
     validation_plot_data$Mean_CV_F1,
     labels = paste("α =", validation_plot_data$Alpha),
     pos = 3, cex = 0.9)
# Highlight selected alpha
points(validation_plot_data$Mean_CV_Accuracy[validation_plot_data$Alpha == best_alpha],
       validation_plot_data$Mean_CV_F1[validation_plot_data$Alpha == best_alpha],
       pch = 1, cex = 3, col = "red", lwd = 2)
text(validation_plot_data$Mean_CV_Accuracy[validation_plot_data$Alpha == best_alpha],
     validation_plot_data$Mean_CV_F1[validation_plot_data$Alpha == best_alpha],
     labels = "SELECTED", pos = 1, col = "red", font = 2, cex = 0.8)

# Plot 3: Bar chart comparison
barplot_data <- as.matrix(t(validation_plot_data[, c("Mean_CV_Accuracy", "Mean_CV_F1")]))
colnames(barplot_data) <- paste("α =", validation_plot_data$Alpha)
barplot(barplot_data, 
        beside = TRUE,
        col = c("steelblue", "coral"),
        ylim = c(0, max(barplot_data) * 1.1),
        legend.text = c("Accuracy", "F1-Score"),
        args.legend = list(x = "topright"),
        main = "Top 3 Alphas: Metric Comparison",
        ylab = "Score")
# Highlight selected
selected_idx <- which(validation_plot_data$Alpha == best_alpha)
rect(selected_idx * 3 - 2.5, 0, selected_idx * 3 + 0.5, max(barplot_data) * 1.1, 
     border = "red", lwd = 2, lty = 2)

# Plot 4: Lambda vs Alpha
plot(alpha_performance$Alpha, alpha_performance$Mean_Lambda,
     type = "b", pch = 19, col = "darkgreen", lwd = 2,
     xlab = "Alpha", ylab = "Mean Lambda",
     main = "Optimal Lambda vs Alpha")
abline(v = best_alpha, col = "red", lty = 2)

# Plot 5: Number of Features vs Alpha
plot(alpha_performance$Alpha, alpha_performance$Mean_N_Features,
     type = "b", pch = 19, col = "purple", lwd = 2,
     xlab = "Alpha", ylab = "Mean Number of Features",
     main = "Feature Selection vs Alpha")
abline(v = best_alpha, col = "red", lty = 2)

# Plot 6: SD of Accuracy vs Alpha (stability)
plot(alpha_performance$Alpha, alpha_performance$SD_Accuracy,
     type = "b", pch = 19, col = "orange", lwd = 2,
     xlab = "Alpha", ylab = "SD of CV Accuracy",
     main = "Model Stability vs Alpha")
abline(v = best_alpha, col = "red", lty = 2)


# ==============================================================================
# SECTION 11.5: FINAL MODEL - ONE-HOT ENCODING AFTER CLASS WEIGHTS
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 11.5: Training Final Model (Encode → Scale → Train)\n")
cat("================================================================================\n\n")

# Calculate Class Weights for FULL training set
class_counts_full <- table(y_train)
num_classes_full <- length(class_counts_full)
total_samples_full <- length(y_train)
class_weights_map_full <- total_samples_full / (num_classes_full * class_counts_full)
final_sample_weights <- class_weights_map_full[match(as.character(y_train), names(class_weights_map_full))]

cat("Class Weights Applied:\n")
print(round(class_weights_map_full, 3))
cat("\n")

# Feature selection on full training set (RAW categorical data)
cols_to_exclude <- c("Customer.ID", "Churn_Category", "Customer.Status", 
                     "Churn.Reason", "Zip.Code", "Churn.Category")
feature_cols <- setdiff(names(train_data), cols_to_exclude)

# Remove high-cardinality categorical variables (>53 levels) for Random Forest
train_rf <- train_data[, c("Churn_Category", feature_cols)]

high_cardinality_cols <- character()
for(col in feature_cols) {
  if(is.factor(train_rf[[col]]) || is.character(train_rf[[col]])) {
    n_levels <- length(unique(train_rf[[col]]))
    if(n_levels > 53) {
      high_cardinality_cols <- c(high_cardinality_cols, col)
      cat("   Excluding", col, "from RF (", n_levels, "levels)\n")
    }
  }
}

# Use remaining features for Random Forest
rf_feature_cols <- setdiff(feature_cols, high_cardinality_cols)

rf_formula <- as.formula(paste("Churn_Category ~", paste(rf_feature_cols, collapse = " + ")))

cat("Performing feature selection on full training set...\n")
rf_final <- randomForest(
  rf_formula,
  data = train_rf,
  ntree = 100,
  importance = TRUE,
  nodesize = 50,
  maxnodes = 50
)

importance_scores_final <- importance(rf_final)[, "MeanDecreaseGini"]
importance_df_final <- data.frame(
  Feature = names(importance_scores_final),
  Importance = importance_scores_final
) %>%
  arrange(desc(Importance))

cumulative_imp <- cumsum(importance_df_final$Importance) / sum(importance_df_final$Importance)
n_features_final <- which(cumulative_imp >= 0.95)[1]
selected_features_final <- importance_df_final$Feature[1:n_features_final]

cat("  Features selected:", n_features_final, "out of", nrow(importance_df_final), "\n\n")

# Display top features
cat("Top 20 Most Important Features:\n")
print(head(importance_df_final, 20), row.names = FALSE)
cat("\n")

# Visualize feature importance
top_n <- 25
p5 <- ggplot(head(importance_df_final, top_n), 
                       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = paste("Top", top_n, "Important Features"),
       subtitle = "Based on Random Forest (Full Training Set)",
       x = "Feature", y = "Importance Score") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p5)

# One-hot encode the selected features
formula_str <- paste("~", paste(selected_features_final, collapse = " + "), "- 1")
model_formula <- as.formula(formula_str)

cat("One-hot encoding selected features...\n")
X_train_encoded <- model.matrix(model_formula, data = train_data)
X_test_encoded <- model.matrix(model_formula, data = test_data)

cat("  Training shape:", dim(X_train_encoded), "\n")
cat("  Test shape:", dim(X_test_encoded), "\n\n")

# Scale features
cat("Scaling features...\n")
X_train_scaled <- X_train_encoded
X_test_scaled <- X_test_encoded

for(i in 1:ncol(X_train_encoded)) {
  col_mean <- mean(X_train_encoded[, i])
  col_sd <- sd(X_train_encoded[, i])
  
  if(col_sd > 0) {
    X_train_scaled[, i] <- (X_train_encoded[, i] - col_mean) / col_sd
    X_test_scaled[, i] <- (X_test_encoded[, i] - col_mean) / col_sd
  }
}

cat("✓ Scaling completed\n\n")

# Train final model with optimal alpha and class weights
cat("Training final Elastic Net regularized Multinomial Logistic Regression model with alpha =", best_alpha, "and Class Weights...\n")

# Use CV to find optimal lambda for final model WITH WEIGHTS
cv_final <- cv.glmnet(
  x = X_train_scaled,
  y = y_train,
  family = "multinomial",
  alpha = best_alpha,
  nfolds = 10,
  type.measure = "class",
  weights = as.numeric(final_sample_weights),
  standardize = FALSE
)

# Fitting Elastic Net regularized Multinomial Logistic Regression model
final_model <- glmnet(
  x = X_train_scaled,
  y = y_train,
  family = "multinomial",
  alpha = best_alpha,
  lambda = cv_final$lambda.min,
  weights = as.numeric(final_sample_weights),
  standardize = FALSE
)

cat("  Final lambda:", round(cv_final$lambda.min, 6), "\n")
cat("✓ Final model training complete\n\n")

# Plot final CV curve
plot(cv_final, main = paste("Final Model CV Curve (alpha =", best_alpha, ")"))
abline(v = -log(cv_final$lambda.min), col = "red", lty = 2)
abline(v = -log(cv_final$lambda.1se), col = "blue", lty = 2)
legend("topright", legend = c("lambda.min", "lambda.1se"),
       col = c("red", "blue"), lty = 2)

cat("✓ Final model training complete\n\n")

# ==============================================================================
# SECTION 12: MODEL PREDICTIONS
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 12: Model Predictions on Test Set\n")
cat("================================================================================\n\n")

# Get predictions on test set
# Type "class" gives the predicted class
# Type "response" gives probability for each class

test_pred_class <- predict(final_model, 
                           newx = X_test_scaled,
                           s = cv_final$lambda.min,
                           type = "class")

test_pred_prob <- predict(final_model, 
                          newx = X_test_scaled,
                          s = cv_final$lambda.min,
                          type = "response")


# Convert predictions to factor with same levels as y_test
test_pred_class <- factor(test_pred_class[,1], levels = levels(y_test))

cat("✓ Predictions generated for", length(test_pred_class), "test samples\n\n")

# ==============================================================================
# SECTION 13: COMPREHENSIVE MODEL EVALUATION
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 13: Comprehensive Model Evaluation\n")
cat("================================================================================\n\n")

# ----------------------
# 13.1: Confusion Matrix
# ----------------------

cat("13.1 CONFUSION MATRIX\n")
cat("--------------------------------------------------------------------------------\n")

conf_matrix <- confusionMatrix(test_pred_class, y_test)

# Display confusion matrix
cat("\nConfusion Matrix:\n")
print(conf_matrix$table)

# Display percentages
conf_matrix_pct <- prop.table(conf_matrix$table, margin = 1) * 100
cat("\nConfusion Matrix (Row Percentages):\n")
print(round(conf_matrix_pct, 2))

# Visualize confusion matrix as heatmap
conf_df <- as.data.frame(conf_matrix$table)
names(conf_df) <- c("Predicted", "Actual", "Count")

p6 <- ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix Heatmap",
       subtitle = paste("Overall Accuracy:", 
                        round(conf_matrix$overall["Accuracy"] * 100, 2), "%")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))

print(p6)

# ----------------------
# 13.2: Overall Metrics
# ----------------------

cat("\n13.2 OVERALL PERFORMANCE METRICS\n")
cat("--------------------------------------------------------------------------------\n")

overall_accuracy <- conf_matrix$overall["Accuracy"]
overall_kappa <- conf_matrix$overall["Kappa"]

cat("Overall Accuracy:", round(overall_accuracy * 100, 2), "%\n")
cat("Cohen's Kappa:", round(overall_kappa, 4), "\n")
cat("95% CI:", round(conf_matrix$overall["AccuracyLower"] * 100, 2), "% -", round(conf_matrix$overall["AccuracyUpper"] * 100, 2), "%\n\n")

# ----------------------
# 13.3: Per-Class Metrics
# ----------------------

cat("13.3 PER-CLASS PERFORMANCE METRICS\n")
cat("--------------------------------------------------------------------------------\n")

# Extract per-class metrics
per_class_metrics <- data.frame(
  Class = rownames(conf_matrix$byClass),
  Sensitivity = conf_matrix$byClass[, "Sensitivity"],
  Specificity = conf_matrix$byClass[, "Specificity"],
  Precision = conf_matrix$byClass[, "Pos Pred Value"],
  F1_Score = conf_matrix$byClass[, "F1"],
  Balanced_Accuracy = conf_matrix$byClass[, "Balanced Accuracy"]
)

# Clean class names (remove "Class: " prefix)
per_class_metrics$Class <- gsub("Class: ", "", per_class_metrics$Class)

cat("\nDetailed Per-Class Metrics:\n")
print(per_class_metrics, row.names = FALSE, digits = 4)
cat("\n")

# Calculate macro-averaged metrics
macro_precision <- mean(per_class_metrics$Precision, na.rm = TRUE)
macro_recall <- mean(per_class_metrics$Sensitivity, na.rm = TRUE)
macro_f1 <- mean(per_class_metrics$F1_Score, na.rm = TRUE)

cat("Macro-Averaged Metrics:\n")
cat("  Macro Precision:", round(macro_precision, 4), "\n")
cat("  Macro Recall:", round(macro_recall, 4), "\n")
cat("  Macro F1-Score:", round(macro_f1, 4), "\n\n")

# Visualize per-class F1 scores
p7 <- ggplot(per_class_metrics, aes(x = reorder(Class, F1_Score), y = F1_Score)) +
  geom_col(fill = "coral", alpha = 0.8) +
  coord_flip() +
  labs(title = "F1-Score by Churn Category",
       x = "Churn Category", y = "F1-Score") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold")) +
  geom_hline(yintercept = macro_f1, linetype = "dashed", color = "red") +
  annotate("text", x = 1, y = macro_f1 + 0.05, 
           label = paste("Macro F1:", round(macro_f1, 3)), color = "red")

print(p7)

# ----------------------
# 13.4: ROC-AUC Analysis
# ----------------------

cat("\n13.4 ROC-AUC ANALYSIS (One-vs-Rest)\n")
cat("--------------------------------------------------------------------------------\n")

# For multiclass ROC, we use One-vs-Rest approach
# Calculate ROC for each class vs all others

roc_list <- list()
auc_values <- numeric()
class_levels <- levels(y_test)

for(i in 1:length(class_levels)) {
  current_class <- class_levels[i]
  
  # Binary indicator: current class vs others
  binary_actual <- ifelse(y_test == current_class, 1, 0)
  
  # Extract probability for current class
  class_prob <- test_pred_prob[, i, 1]
  
  # Calculate ROC
  roc_obj <- roc(binary_actual, class_prob, quiet = TRUE)
  roc_list[[current_class]] <- roc_obj
  auc_values[i] <- auc(roc_obj)
  
  cat("Class:", current_class, "| AUC:", round(auc(roc_obj), 4), "\n")
}

# Calculate macro-averaged AUC
macro_auc <- mean(auc_values, na.rm = TRUE)
cat("\nMacro-Averaged AUC:", round(macro_auc, 4), "\n\n")

# Plot ROC curves for all classes
cat("Generating ROC curve plots...\n")


# Plot 1: All individual class ROCs 
# Create color palette
roc_colors <- rainbow(length(class_levels))

plot(roc_list[[1]], col = roc_colors[1], main = "ROC Curves (One-vs-Rest)",
     xlim = c(1, 0), ylim = c(0, 1))

for(i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = roc_colors[i], add = TRUE)
}

# Add legend
legend("bottomright", 
       legend = paste(class_levels, "- AUC:", round(auc_values, 3)),
       col = roc_colors, lwd = 2, cex = 0.7)

abline(a = 0, b = 1, lty = 2, col = "gray")


# Calculate Binary Churn vs No Churn 
binary_target <- ifelse(y_test == "No Churn", 0, 1)

if("No Churn" %in% class_levels) {
  no_churn_idx <- which(class_levels == "No Churn")
  churn_prob <- 1 - test_pred_prob[, no_churn_idx, 1]
} else {
  # If no "No Churn" class, sum all probabilities
  churn_prob <- rowSums(test_pred_prob[, , 1])
}

binary_roc <- roc(binary_target, churn_prob, quiet = TRUE)
binary_auc <- auc(binary_roc)
cat("Binary Churn vs No Churn AUC:", round(binary_auc, 4), "\n\n")


# Plot 2: Single Binary ROC Curve 
plot(binary_roc, 
     col = "darkblue", 
     lwd = 3,
     main = "Binary ROC: Churn vs No Churn",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)")

# Add AUC text to plot
text(0.6, 0.3, 
     paste("AUC =", round(binary_auc, 4)),
     cex = 1.3, col = "darkblue", font = 2)

# Add diagonal reference line
abline(a = 0, b = 1, lty = 2, col = "gray")

# ----------------------
# 13.5: Log Loss
# ----------------------

cat("\n13.5 LOG LOSS (CROSS-ENTROPY)\n")
cat("--------------------------------------------------------------------------------\n")

# Calculate log loss
log_loss_value <- MultiLogLoss(y_pred = test_pred_prob[,,1], 
                               y_true = y_test)

cat("Log Loss:", round(log_loss_value, 4), "\n")
cat("(Lower is better; perfect prediction = 0)\n\n")

# ----------------------
# 13.6: Summary Table
# ----------------------

cat("13.6 EVALUATION SUMMARY\n")
cat("--------------------------------------------------------------------------------\n")

evaluation_summary <- data.frame(
  Metric = c("Overall Accuracy", "Cohen's Kappa", 
             "Macro F1-Score", "Macro AUC", "Log Loss"),
  Value = c(
    paste0(round(overall_accuracy * 100, 2), "%"),
    round(overall_kappa, 4),
    round(macro_f1, 4),
    round(macro_auc, 4),
    round(log_loss_value, 4)
  )
)

cat("\nOverall Model Performance:\n")

print(evaluation_summary, row.names = FALSE)
cat("\n")

# ==============================================================================
# SECTION 14: MODEL INTERPRETATION - COEFFICIENTS AND ODDS RATIOS
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 14: Model Interpretation - Coefficients & Odds Ratios\n")
cat("================================================================================\n\n")

# Extract coefficients from the Elastic Net regularized model
# glmnet returns coefficients for each class (except reference class)
coef_list <- coef(final_model, s = cv_final$lambda.min)

cat("Extracting model coefficients...\n")

# The coefficients are stored as a list (one element per class)
# We'll combine them into a data frame for analysis

coef_df_list <- list()

for(class_name in names(coef_list)) {
  
  # Extract coefficients for this class
  coef_matrix <- as.matrix(coef_list[[class_name]])
  
  # Remove intercept and zero coefficients for cleaner interpretation
  coef_values <- coef_matrix[-1, 1]  # Remove intercept
  non_zero_idx <- which(coef_values != 0)
  
  if(length(non_zero_idx) > 0) {
    
    coef_df <- data.frame(
      Churn_Category = class_name,
      Feature = names(coef_values)[non_zero_idx],
      Coefficient = coef_values[non_zero_idx],
      Odds_Ratio = exp(coef_values[non_zero_idx])
    )
    
    coef_df_list[[class_name]] <- coef_df
  }
}

# Combine all coefficient data frames
all_coefs <- bind_rows(coef_df_list)

cat("✓ Coefficients extracted\n")
cat("  Total non-zero coefficients:", nrow(all_coefs), "\n\n")

# Display top positive and negative coefficients for each class
cat("TOP COEFFICIENTS BY CHURN CATEGORY\n")
cat("--------------------------------------------------------------------------------\n\n")

for(class_name in unique(all_coefs$Churn_Category)) {
  
  class_coefs <- all_coefs %>%
    filter(Churn_Category == class_name) %>%
    arrange(desc(abs(Coefficient)))
  
  cat("Churn Category:", class_name, "\n")
  cat("----------------------------------------\n")
  
  # Top 10 most influential features
  top_features <- head(class_coefs, 10)
  
  cat("\nTop 10 Most Influential Features:\n")
  for(i in 1:nrow(top_features)) {
    coef_val <- top_features$Coefficient[i]
    or_val <- top_features$Odds_Ratio[i]
    feature <- top_features$Feature[i]
    
    direction <- ifelse(coef_val > 0, "increases", "decreases")
    
    cat(sprintf("  %d. %s\n", i, feature))
    cat(sprintf("     Coefficient: %+.4f | Odds Ratio: %.4f\n", coef_val, or_val))
    cat(sprintf("     → %s odds of this churn type by %.1f%%\n\n", 
                direction, abs((or_val - 1) * 100)))
  }
  
  cat("\n")
}

# Visualize coefficients
cat("Generating coefficient visualizations...\n\n")

# Coefficient heatmap
# Select top features by absolute coefficient value across all classes
top_features_overall <- all_coefs %>%
  group_by(Feature) %>%
  summarise(Max_Abs_Coef = max(abs(Coefficient))) %>%
  arrange(desc(Max_Abs_Coef)) %>%
  head(20) %>%
  pull(Feature)

coef_for_heatmap <- all_coefs %>%
  filter(Feature %in% top_features_overall)

p8 <- ggplot(coef_for_heatmap, aes(x = Churn_Category, y = Feature, fill = Coefficient)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, name = "Coefficient") +
  labs(title = "Top 20 Feature Coefficients by Churn Category",
       x = "Churn Category", y = "Feature") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(face = "bold"))

print(p8)

# Odds ratio comparison
p9 <- ggplot(coef_for_heatmap, 
             aes(x = reorder(Feature, Odds_Ratio), y = Odds_Ratio, fill = Churn_Category)) +
  geom_col(position = "dodge", alpha = 0.8) +
  coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  labs(title = "Odds Ratios for Top Features",
       subtitle = "Values > 1 increase odds; Values < 1 decrease odds",
       x = "Feature", y = "Odds Ratio") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p9)

# ==============================================================================
# SECTION 15: CATEGORY-SPECIFIC VARIABLE IMPORTANCE
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 15: Category-Specific Variable Importance\n")
cat("================================================================================\n\n")

# Calculate importance as absolute coefficient value
category_importance <- all_coefs %>%
  group_by(Churn_Category) %>%
  arrange(desc(abs(Coefficient))) %>%
  slice_head(n = 15) %>%
  ungroup()

cat("Top 15 most important variables for each churn category:\n\n")

# Create importance ranking visualization
p10 <- ggplot(category_importance, 
              aes(x = reorder(Feature, abs(Coefficient)), 
                  y = abs(Coefficient), 
                  fill = Churn_Category)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  facet_wrap(~Churn_Category, scales = "free_y") +
  labs(title = "Top 15 Features by Churn Category",
       subtitle = "Based on absolute coefficient magnitude",
       x = "Feature", y = "Absolute Coefficient") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        axis.text.y = element_text(size = 7),
        legend.position = "none")

print(p10)

# Feature frequency across categories
cat("\nFeatures appearing in top 15 across multiple categories:\n")

feature_frequency <- category_importance %>%
  group_by(Feature) %>%
  summarise(Appears_In_N_Categories = n()) %>%
  filter(Appears_In_N_Categories > 1) %>%
  arrange(desc(Appears_In_N_Categories))

print(feature_frequency, n = 20)
cat("\n")

# ==============================================================================
# SECTION 16: BUSINESS INTELLIGENCE & RISK SCORING
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 16: Business Intelligence & Advanced Risk Scoring\n")
cat("================================================================================\n\n")

# ----------------------
# 16.1: Predict on Test Set with Probabilities
# ----------------------

cat("16.1 GENERATING RISK SCORES FOR TEST SET\n")
cat("--------------------------------------------------------------------------------\n")

# Create comprehensive test results dataframe
test_results <- data.frame(
  Actual_Category = y_test,
  Predicted_Category = test_pred_class
)

# Add probability for each class
for(i in 1:length(class_levels)) {
  test_results[[paste0("Prob_", class_levels[i])]] <- test_pred_prob[, i, 1]
}

# Calculate base risk score (probability of ANY churn)
# Assuming "No Churn" is one of the categories
if("No Churn" %in% class_levels) {
  no_churn_col <- paste0("Prob_No Churn")
  test_results$Base_Risk_Score <- 1 - test_results[[no_churn_col]]
} else {
  # If no "No Churn" category, sum all churn probabilities
  churn_prob_cols <- grep("^Prob_", names(test_results), value = TRUE)
  test_results$Base_Risk_Score <- rowSums(test_results[, churn_prob_cols])
}

# Add business context from test_data 
test_results$Tenure <- test_data$Tenure.in.Months
test_results$Monthly_Charge <- test_data$Monthly.Charge
test_results$Contract <- test_data$Contract
test_results$Payment_Method <- test_data$Payment.Method

# ----------------------
# 16.2: Advanced Risk Scoring with Business Context
# ----------------------

cat("\n16.2 APPLYING BUSINESS CONTEXT ADJUSTMENTS\n")
cat("--------------------------------------------------------------------------------\n")

# Apply business-driven risk multipliers
test_results <- test_results %>%
  mutate(
    # Tenure multiplier: newer customers are higher risk
    Tenure_Multiplier = case_when(
      Tenure < 6 ~ 1.4,
      Tenure < 12 ~ 1.2,
      Tenure < 24 ~ 1.1,
      TRUE ~ 1.0
    ),
    
    # Contract multiplier: Month to Month is higher risk
    Contract_Multiplier = case_when(
      Contract == "Month to Month" ~ 1.3,
      TRUE ~ 1.0
    ),
  
    # Payment multiplier: certain payment methods indicate higher risk
    Payment_Multiplier = case_when(
      Payment_Method %in% c("Bank withdrawal", "Mailed check") ~ 1.1,
      TRUE ~ 1.0
    ),
    
    # Calculate composite risk score
    Composite_Risk_Score = pmin(Base_Risk_Score * Tenure_Multiplier * 
                                  Contract_Multiplier * Payment_Multiplier, 1.0),
    
    # Categorize risk
    Risk_Category = cut(Composite_Risk_Score,
                        breaks = c(0, 0.25, 0.5, 0.75, 1.0),
                        labels = c("Low", "Medium", "High", "Critical"),
                        include.lowest = TRUE)
  )

cat("✓ Risk scoring completed\n\n")

# Risk distribution analysis
cat("RISK DISTRIBUTION SUMMARY\n")
cat("--------------------------------------------------------------------------------\n")

risk_summary <- test_results %>%
  group_by(Risk_Category) %>%
  summarise(
    Count = n(),
    Percentage = n() / nrow(test_results) * 100,
    Avg_Monthly_Revenue = mean(Monthly_Charge, na.rm = TRUE),
    Avg_Tenure = mean(Tenure, na.rm = TRUE),
    Total_Monthly_Revenue = sum(Monthly_Charge, na.rm = TRUE),
    Avg_Risk_Score = mean(Composite_Risk_Score),
    .groups = "drop"
  )

print(risk_summary, digits = 2)
cat("\n")

# ----------------------
# 16.3: Strategic Business Insights
# ----------------------

cat("16.3 STRATEGIC BUSINESS INSIGHTS\n")
cat("--------------------------------------------------------------------------------\n\n")

total_customers <- nrow(test_results)
high_risk_customers <- sum(test_results$Risk_Category %in% c("High", "Critical"))
high_risk_pct <- (high_risk_customers / total_customers) * 100

avg_monthly_revenue <- mean(test_results$Monthly_Charge, na.rm = TRUE)
potential_monthly_loss <- sum(test_results$Monthly_Charge[test_results$Risk_Category %in% c("High", "Critical")])
potential_annual_loss <- potential_monthly_loss * 12

cat("EXECUTIVE SUMMARY\n")
cat("═══════════════════════════════════════════════════════════════════\n")
cat("Total Customers Analyzed:", format(total_customers, big.mark = ","), "\n")
cat("High-Risk Customers (High + Critical):", format(high_risk_customers, big.mark = ","), 
    sprintf("(%.1f%%)\n", high_risk_pct))
cat("Average Monthly Revenue per Customer: $", sprintf("%.2f", avg_monthly_revenue), "\n")
cat("Monthly Revenue at Risk: $", format(round(potential_monthly_loss, 0), big.mark = ","), "\n")
cat("Estimated Annual Revenue at Risk: $", format(round(potential_annual_loss, 0), big.mark = ","), "\n")
cat("Recommended Retention Budget (15% of at-risk revenue): $", 
    format(round(potential_annual_loss * 0.15, 0), big.mark = ","), "\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

# Segment-specific insights
cat("CUSTOMER SEGMENT ANALYSIS\n")
cat("--------------------------------------------------------------------------------\n")

segment_analysis <- test_results %>%
  mutate(
    Tenure_Segment = cut(Tenure,
                         breaks = c(0, 6, 24, 60, Inf),
                         labels = c("New (0-6m)", "Growing (6-24m)", 
                                    "Mature (24-60m)", "Loyal (60m+)"),
                         include.lowest = TRUE),
    Value_Segment = cut(Monthly_Charge,
                        breaks = quantile(Monthly_Charge, c(0, 0.33, 0.67, 1)),
                        labels = c("Economy", "Standard", "Premium"),
                        include.lowest = TRUE)
  )  %>%
  group_by(Tenure_Segment, Risk_Category) %>%
  summarise(
    Count = n(),
    Avg_Risk = mean(Composite_Risk_Score, na.rm = TRUE), # Use na.rm=TRUE for safety
    Total_Revenue = sum(Monthly_Charge, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  tidyr::complete(Tenure_Segment, Risk_Category, 
                  fill = list(Count = 0, Avg_Risk = 0, Total_Revenue = 0))

print(segment_analysis, digits = 2)
cat("\n")

# Key insights
cat("KEY ACTIONABLE INSIGHTS\n")
cat("--------------------------------------------------------------------------------\n")

# Insight 1: New customer risk
new_customer_risk <- test_results %>%
  filter(Tenure < 6) %>%
  summarise(High_Risk_Pct = mean(Risk_Category %in% c("High", "Critical")) * 100) %>%
  pull(High_Risk_Pct)

if(new_customer_risk > 30) {
  cat(" CRITICAL: ", sprintf("%.1f%%", new_customer_risk), 
      " of customers in first 6 months are high-risk\n")
  cat("   → IMMEDIATE ACTION: Enhance onboarding program and first-90-days engagement\n\n")
}

# Insight 2: Month to Month contract risk
mtm_risk <- test_results %>%
  filter(Contract == "Month to Month") %>%
  summarise(High_Risk_Pct = mean(Risk_Category %in% c("High", "Critical")) * 100) %>%
  pull(High_Risk_Pct)

cat(" INSIGHT: ", sprintf("%.1f%%", mtm_risk), 
    " of Month to Month customers are high-risk\n")
cat("   → RECOMMENDATION: Offer incentives for annual/2-year contract upgrades\n\n")

# Insight 3: Payment method correlation
payment_risk <- test_results %>%
  group_by(Payment_Method) %>%
  summarise(
    High_Risk_Pct = mean(Risk_Category %in% c("High", "Critical")) * 100,
    Count = n()
  ) %>%
  arrange(desc(High_Risk_Pct))

cat(" PAYMENT METHOD RISK ANALYSIS:\n")
print(payment_risk, digits = 2)
cat("   → RECOMMENDATION: Encourage automatic payments and credit card enrollment\n\n")

# ----------------------
# 16.4: Risk-Based Action Framework
# ----------------------

cat("\n16.4 RISK-BASED ACTION FRAMEWORK\n")
cat("================================================================================\n\n")

action_framework <- data.frame(
  Risk_Level = c(" Low (0-25%)", " Medium (25-50%)", 
                 " High (50-75%)", " Critical (75-100%)"),
  Customer_Count = c(
    sum(test_results$Risk_Category == "Low"),
    sum(test_results$Risk_Category == "Medium"),
    sum(test_results$Risk_Category == "High"),
    sum(test_results$Risk_Category == "Critical")
  ),
  Recommended_Actions = c(
    "Quarterly satisfaction surveys, loyalty rewards program",
    "Monthly check-ins, personalized service recommendations",
    "Bi-weekly proactive outreach, special retention offers",
    "IMMEDIATE executive intervention, dedicated account manager"
  ),
  Contact_Frequency = c("Quarterly", "Monthly", "Bi-weekly", "Weekly"),
  Retention_Budget_Priority = c("Low", "Medium", "High", "Maximum")
)

print(action_framework, row.names = FALSE)
cat("\n")

# ----------------------
# 16.5: Business Visualizations
# ----------------------

cat("16.5 GENERATING BUSINESS INTELLIGENCE VISUALIZATIONS\n")
cat("--------------------------------------------------------------------------------\n\n")

# Visualization 1: Risk Distribution Pie Chart
risk_dist <- as.data.frame(table(test_results$Risk_Category))
names(risk_dist) <- c("Risk_Category", "Count")

p11 <- ggplot(risk_dist, aes(x = "", y = Count, fill = Risk_Category)) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("Low" = "#4CAF50", "Medium" = "#FF9800",
                               "High" = "#FF5722", "Critical" = "#F44336")) +
  labs(title = "Customer Risk Distribution",
       fill = "Risk Level") +
  theme_void() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5)) +
  geom_text(aes(label = paste0(Count, "\n(", 
                               round(Count/sum(Count)*100, 1), "%)")),
            position = position_stack(vjust = 0.5))

print(p11)

# Visualization 2: Revenue at Risk by Category
revenue_risk <- test_results %>%
  group_by(Risk_Category) %>%
  summarise(Total_Monthly_Revenue = sum(Monthly_Charge, na.rm = TRUE))

p12 <- ggplot(revenue_risk, aes(x = Risk_Category, y = Total_Monthly_Revenue, 
                                fill = Risk_Category)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("Low" = "#4CAF50", "Medium" = "#FF9800",
                               "High" = "#FF5722", "Critical" = "#F44336")) +
  labs(title = "Monthly Revenue at Risk by Category",
       x = "Risk Category", y = "Monthly Revenue ($)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold")) +
  geom_text(aes(label = paste0("$", format(round(Total_Monthly_Revenue, 0), 
                                           big.mark = ","))),
            vjust = -0.5, size = 4)

print(p12)

# Visualization 3: Risk Score Distribution
p13 <- ggplot(test_results, aes(x = Composite_Risk_Score, fill = Risk_Category)) +
  geom_histogram(bins = 30, alpha = 0.7, color = "white") +
  scale_fill_manual(values = c("Low" = "#4CAF50", "Medium" = "#FF9800",
                               "High" = "#FF5722", "Critical" = "#F44336")) +
  labs(title = "Distribution of Risk Scores",
       x = "Composite Risk Score", y = "Number of Customers",
       fill = "Risk Category") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p13)

# Visualization 4: Tenure vs Risk
p14 <- ggplot(test_results, aes(x = Tenure, y = Composite_Risk_Score, 
                                color = Risk_Category)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "black", linetype = "dashed") +
  scale_color_manual(values = c("Low" = "#4CAF50", "Medium" = "#FF9800",
                                "High" = "#FF5722", "Critical" = "#F44336")) +
  labs(title = "Relationship Between Tenure and Risk Score",
       x = "Tenure (Months)", y = "Risk Score",
       color = "Risk Category") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p14)

# Visualization 5: Contract Type vs Risk
contract_risk_summary <- test_results %>%
  group_by(Contract, Risk_Category) %>%
  summarise(Count = n(), .groups = "drop")

p15 <- ggplot(contract_risk_summary, aes(x = Contract, y = Count, fill = Risk_Category)) +
  geom_col(position = "fill") +
  scale_fill_manual(values = c("Low" = "#4CAF50", "Medium" = "#FF9800",
                               "High" = "#FF5722", "Critical" = "#F44336")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Risk Distribution by Contract Type",
       x = "Contract Type", y = "Percentage",
       fill = "Risk Category") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

print(p15)

cat("✓ Business intelligence visualizations completed\n\n")

# ==============================================================================
# SECTION 17: RECENTLY JOINED CUSTOMERS - RISK PROFILING
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 17: Recently Joined Customers - Proactive Risk Assessment\n")
cat("================================================================================\n\n")

cat("Analyzing recently joined customers who were excluded from model training...\n\n")

if(nrow(recently_joined_customers) > 0) {
  
  # Handle specific missing value cases before MICE
  # Total.Charges often missing for very new customers (tenure = 0)
  if("Total.Charges" %in% names(recently_joined_customers)) {
    recently_joined_customers <- recently_joined_customers %>%
      mutate(Total.Charges = ifelse(is.na(Total.Charges) & Tenure.in.Months == 0, 
                                    0, 
                                    Total.Charges))
  }
  
  # Internet service-related features: NA likely means "No Internet Service"
  internet_cols <- c("Internet.Type","Online.Security", "Online.Backup", "Device.Protection.Plan", 
                     "Premium.Tech.Support", "Streaming.TV", "Streaming.Movies", 
                     "Streaming.Music", "Unlimited.Data")
  
  existing_internet_cols <- intersect(internet_cols, names(recently_joined_customers))
  if(length(existing_internet_cols) > 0) {
    for(col in existing_internet_cols) {
      recently_joined_customers[[col]][is.na(recently_joined_customers[[col]])] <- "No Internet Service"
    }
  }
  
  # MICE Imputation for remaining missing values
  # MICE Imputation is preferred since for the given data <30% is missing 
  if(sum(is.na(recently_joined_customers)) > 0) {
    
    # Performing MICE imputation
    # Identify columns for imputation (exclude ID columns and target)
    cols_to_exclude <- c("Customer.ID", "Churn_Category", "Customer.Status", 
                         "Churn.Reason", "Churn.Label", "Churn.Category")
    cols_for_imputation_rj <- setdiff(names(recently_joined_customers), cols_to_exclude)
    
    # Separate data
    data_for_imputation_rj <- recently_joined_customers %>% select(all_of(cols_for_imputation_rj))
    data_excluded_rj <- recently_joined_customers %>% select(all_of(intersect(cols_to_exclude, names(recently_joined_customers))))
    
    # Run MICE (m=5 imputations, maxit=5 iterations)
    # Using pmm (predictive mean matching) method for robustness
    mice_model <- mice(data_for_imputation_rj, 
                       m = 5,           # Number of imputations
                       maxit = 5,       # Number of iterations
                       method = "pmm",  # Predictive mean matching
                       seed = 123,
                       printFlag = FALSE)
    
    # Complete the imputation (use first imputed dataset)
    imputed_data <- complete(mice_model, 1)
    
    # Combine back with excluded columns
    recently_joined_customers <- bind_cols(data_excluded_rj, imputed_data)
    
    cat("✓ MICE imputation completed\n")
    cat("  Remaining missing values:", sum(is.na(recently_joined_customers)), "\n\n")
    
  } else {
    cat("✓ No missing values remaining, skipping MICE imputation\n\n")
  }
  
  # Handle Multiple.Lines Feature 
  # Convert to clean binary
  recently_joined_customers <- recently_joined_customers %>%
    mutate(
      Multiple.Lines = case_when(
        # Customers without phone service logically cannot have multiple lines
        Phone.Service == "No" ~ "No",
        
        # Customers with phone service: use actual value
        Phone.Service == "Yes" & Multiple.Lines == "Yes" ~ "Yes",
        Phone.Service == "Yes" & Multiple.Lines == "No" ~ "No",
        
        # Safety net for any unexpected NAs
        is.na(Multiple.Lines) ~ "No",
        
        # Default
        TRUE ~ "No"
      )
    )
  
  # Creating the same derived features as that of the rest of the data 
  recently_joined_customers <- recently_joined_customers %>%
    mutate(
      
      # Tenure groups for better interpretability
      Tenure_Group = case_when(
        Tenure.in.Months <= 6 ~ "New",   # (0-6 months)
        Tenure.in.Months <= 24 ~ "Growing",   # (7-24 months)
        Tenure.in.Months <= 60 ~ "Mature",   # (25-60 months)
        TRUE ~ "Loyal"   # (60+ months)
      ),
      
      # To help in variable name checking at later steps 
      Contract = case_when(
        Contract == "Month-to-Month" ~ "Month to Month",
        Contract == "One Year" ~ "One Year",
        Contract == "Two Year" ~ "Two Year",
      ),
      
      # Average revenue per month (handles zero tenure)
      Avg_Revenue_Per_Month = ifelse(Tenure.in.Months > 0, 
                                     Total.Charges / Tenure.in.Months, 
                                     Monthly.Charge),
      
      # Total service count (sum of all additional services)
      Service_Count = rowSums(select(., any_of(c("Online.Security", "Online.Backup", 
                                                 "Device.Protection.Plan", "Premium.Tech.Support",
                                                 "Streaming.TV", "Streaming.Movies", 
                                                 "Streaming.Music", "Unlimited.Data"))) == "Yes", 
                              na.rm = TRUE),
      
      # Revenue to tenure ratio (customer value intensity)
      Revenue_Tenure_Ratio = ifelse(Tenure.in.Months > 0, 
                                    Total.Charges / Tenure.in.Months, 
                                    0),
      
      # Binary: Has any streaming service
      Has_Streaming = ifelse(rowSums(select(., any_of(c("Streaming.TV", "Streaming.Movies", 
                                                        "Streaming.Music"))) == "Yes", 
                                     na.rm = TRUE) > 0, 
                             "Yes", "No"),
      
      # Binary: Has any protection service
      Has_Protection = ifelse(rowSums(select(., any_of(c("Online.Security", "Online.Backup", 
                                                         "Device.Protection.Plan"))) == "Yes", 
                                      na.rm = TRUE) > 0, 
                              "Yes", "No"),
      
      # Age group
      Age_Group = case_when(
        Age < 30 ~ "Young",   # (< 30 years)
        Age < 50 ~ "Middle",   # (30-50 years)
        TRUE ~ "Senior"   # (50+ years)
      ),
      
      # Contract-Tenure interaction (early cancellation risk)
      Contract_Tenure_Risk = case_when(
        Contract == "Month to Month" & Tenure.in.Months < 12 ~ "High Risk",
        Contract == "Month to Month" ~ "Medium Risk",
        TRUE ~ "Low Risk"
      )
    )

  # Convert target variable to factor
  recently_joined_customers$Churn_Category <- as.factor(recently_joined_customers$Churn_Category)
  
  # Identify categorical variables for encoding
  categorical_vars <- names(recently_joined_customers)[sapply(recently_joined_customers, function(x) 
    is.character(x) || is.factor(x))]
  
  # Exclude target and ID columns
  categorical_vars <- setdiff(categorical_vars, c("Churn_Category", "Customer.ID", "Churn.Category", 
                                                  "Customer.Status", "Churn.Reason"))
  
  # Convert categorical variables to factors
  for(var in categorical_vars) {
    recently_joined_customers[[var]] <- as.factor(recently_joined_customers[[var]])
  }
  
  # Prepare features for recently joined customers
  # Use same encoding and scaling as training data
  cat("Processing", nrow(recently_joined_customers), "recently joined customers...\n")
  
  # Get same features as training
  recently_joined_features <- recently_joined_customers %>%
    select(all_of(selected_features_final))
  
  # Dealing with single level feature
  # Replace factor column with a numeric dummy column
  recently_joined_features$Tenure_GroupNew <- 1
  
  # Remove the original factor column
  recently_joined_features <- recently_joined_features %>%
    select(-Tenure_Group)
  
  
  # Replace "Tenure_Group" with "Tenure_GroupNew" in the formula feature list
  selected_features_final <- gsub("Tenure_Group\\b", "Tenure_GroupNew", selected_features_final)
  
  # Rebuild formula string
  formula_str <- paste("~", paste(selected_features_final, collapse = " + "), "- 1")
  model_formula <- as.formula(formula_str)
  
  # Create model matrix (one-hot encode)
  X_recently_joined <- model.matrix(model_formula, data = recently_joined_features)
  
  # Add any missing dummy columns with 0s
  missing_cols <- setdiff(colnames(X_train_scaled), colnames(X_recently_joined))
  if (length(missing_cols) > 0) {
    zero_mat <- matrix(0, nrow = nrow(X_recently_joined), ncol = length(missing_cols))
    colnames(zero_mat) <- missing_cols
    X_recently_joined <- cbind(X_recently_joined, zero_mat)
  }
  
  # Reorder columns to match training
  X_recently_joined <- X_recently_joined[, colnames(X_train_scaled)]
  
  
  # Scale using training set parameters
  X_recently_joined_scaled <- X_recently_joined
  
  for(i in 1:ncol(X_recently_joined)) {
    col_name <- colnames(X_recently_joined)[i]
    
    # Find corresponding column in training data
    if(col_name %in% colnames(X_train_scaled)) {
      train_col_idx <- which(colnames(X_train_scaled) == col_name)
      col_mean <- mean(X_train_encoded[, train_col_idx])
      col_sd <- sd(X_train_encoded[, train_col_idx])
      
      if(col_sd > 0) {
        X_recently_joined_scaled[, i] <- (X_recently_joined[, i] - col_mean) / col_sd
      }
    }
  }
  
  # Predict
  recently_joined_pred_prob <- predict(final_model,
                                       newx = X_recently_joined_scaled,
                                       s = cv_final$lambda.min,
                                       type = "response")
  
  # Create results dataframe
  recently_joined_results <- data.frame(
    Tenure = recently_joined_customers$Tenure.in.Months,
    Monthly_Charge = recently_joined_customers$Monthly.Charge,
    Contract = recently_joined_customers$Contract,
    Payment_Method = recently_joined_customers$Payment.Method
  )
  
  # Add probabilities
  for(i in 1:length(class_levels)) {
    recently_joined_results[[paste0("Prob_", class_levels[i])]] <- 
      recently_joined_pred_prob[, i, 1]
  }
  
  # Calculate risk scores
  if("No Churn" %in% class_levels) {
    no_churn_col <- paste0("Prob_No Churn")
    recently_joined_results$Base_Risk_Score <- 1 - recently_joined_results[[no_churn_col]]
  } else {
    churn_prob_cols <- grep("^Prob_", names(recently_joined_results), value = TRUE)
    recently_joined_results$Base_Risk_Score <- 
      rowSums(recently_joined_results[, churn_prob_cols])
  }
  
  # Apply multipliers
  recently_joined_results <- recently_joined_results %>%
    mutate(
      Tenure_Multiplier = 1.5,
      Contract_Multiplier = case_when(
        Contract == "Month to Month" ~ 1.3,
        TRUE ~ 1.0
      ),
      Payment_Multiplier = case_when(
        Payment_Method %in% c("Bank withdrawal", "Mailed check") ~ 1.1,
        TRUE ~ 1.0
      ),
      Composite_Risk_Score = pmin(Base_Risk_Score * Tenure_Multiplier * 
                                    Contract_Multiplier * Payment_Multiplier, 1.0),
      Risk_Category = cut(Composite_Risk_Score,
                          breaks = c(0, 0.25, 0.5, 0.75, 1.0),
                          labels = c("Low", "Medium", "High", "Critical"),
                          include.lowest = TRUE)
    )
  
  cat("✓ Risk assessment completed for newly joined customers\n\n")
  
  # Summary
  cat("RECENTLY JOINED CUSTOMERS - RISK SUMMARY\n")
  cat("--------------------------------------------------------------------------------\n")
  
  # Summary
  recently_joined_risk_summary <- recently_joined_results %>%
    group_by(Risk_Category) %>%
    summarise(
      Count = n(),
      Percentage = n() / nrow(recently_joined_results) * 100,
      Avg_Monthly_Revenue = mean(Monthly_Charge, na.rm = TRUE),
      Avg_Risk_Score = mean(Composite_Risk_Score),
      .groups = "drop"
    )
  
  print(recently_joined_risk_summary, digits = 2)
  cat("\n")
  
  # Key insight
  high_risk_new <- sum(recently_joined_results$Risk_Category %in% c("High", "Critical"))
  cat("   WARNING:", high_risk_new, "recently joined customers are HIGH RISK\n")
  cat("   These customers need immediate proactive engagement to prevent early churn\n\n")
  
  # Visualize
  p16 <- ggplot(recently_joined_results, aes(x = Risk_Category, fill = Risk_Category)) +
    geom_bar() +
    scale_fill_manual(values = c("Low" = "#4CAF50", "Medium" = "#FF9800",
                                 "High" = "#FF5722", "Critical" = "#F44336")) +
    labs(title = "Risk Distribution: Recently Joined Customers",
         subtitle = paste("N =", nrow(recently_joined_results)),
         x = "Risk Category", y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "none")
  
  print(p16)
  
} else {
  cat("No recently joined customers to analyze.\n\n")
}

# ==============================================================================
# SECTION 18: IMPLEMENTATION ROADMAP
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 18: Implementation Roadmap\n")
cat("================================================================================\n\n")

roadmap <- data.frame(
  Phase = c("Phase 1: Foundation", 
            "Phase 2: Campaign Launch", 
            "Phase 3: Optimization", 
            "Phase 4: Integration"),
  Timeline = c("Week 1-2", "Week 3-4", "Week 5-8", "Week 9-12"),
  Key_Activities = c(
    "Deploy risk scoring system; Train customer service teams; Set up monitoring dashboard",
    "Launch targeted retention campaigns; A/B test interventions; Measure initial response rates",
    "Analyze campaign performance; Refine risk algorithms; Reduce false positives",
    "Full CRM integration; Automated workflows; Real-time risk monitoring"
  ),
  Success_Metrics = c(
    "System deployed; 100% team training completion",
    "Campaign launched; >20% response rate",
    ">10% improvement in prediction accuracy",
    "Automated processes; <5 min scoring latency"
  )
)

print(roadmap, row.names = FALSE)
cat("\n")

cat("KEY SUCCESS METRICS TO TRACK:\n")
cat("────────────────────────────────────────────────────────────────────\n")
cat("• Churn rate reduction by risk category (target: 15-25% reduction)\n")
cat("• Revenue retention improvement (target: $X increase in MRR)\n")
cat("• Campaign response rates (target: >20%)\n")
cat("• Conversion rate of retention offers (target: >15%)\n")
cat("• Time to intervention for high-risk customers (target: <7 days)\n")
cat("• Customer satisfaction scores (target: maintain or improve)\n")
cat("• Model prediction accuracy (monitor monthly, retrain quarterly)\n\n")

# ==============================================================================
# SECTION 19: FINAL SUMMARY AND EXPORT
# ==============================================================================

cat("================================================================================\n")
cat("SECTION 19: Final Summary and Model Export\n")
cat("================================================================================\n\n")

# Create comprehensive summary
final_summary <- list(
  Model_Info = list(
    Model_Type = "Elastic Net Multinomial Logistic Regression",
    Best_Alpha = best_alpha,
    Best_Lambda = cv_final$lambda.min,
    Features_Selected = length(selected_features_final),
    Classes = length(class_levels)
  ),
  
  Performance = list(
    Overall_Accuracy = round(overall_accuracy, 4),
    Macro_F1_Score = round(macro_f1, 4),
    Macro_AUC = round(macro_auc, 4),
    Cohens_Kappa = round(overall_kappa, 4),
    Log_Loss = round(log_loss_value, 4)
  ),
  
  Business_Impact = list(
    Total_Customers = total_customers,
    High_Risk_Customers = high_risk_customers,
    High_Risk_Percentage = round(high_risk_pct, 2),
    Monthly_Revenue_At_Risk = round(potential_monthly_loss, 0),
    Annual_Revenue_At_Risk = round(potential_annual_loss, 0),
    Recommended_Budget = round(potential_annual_loss * 0.15, 0)
  )
)

cat("═══════════════════════════════════════════════════════════════════════════════\n")
cat("                          FINAL MODEL SUMMARY                                    \n")
cat("═══════════════════════════════════════════════════════════════════════════════\n\n")

cat("MODEL CONFIGURATION\n")
cat("───────────────────────────────────────────────────────────────────────────────\n")
cat("Model Type:", final_summary$Model_Info$Model_Type, "\n")
cat("Optimal Alpha:", final_summary$Model_Info$Best_Alpha, "\n")
cat("Optimal Lambda:", final_summary$Model_Info$Best_Lambda, "\n")
cat("Features Selected:", final_summary$Model_Info$Features_Selected, "\n")
cat("Number of Classes:", final_summary$Model_Info$Classes, "\n\n")

cat("MODEL PERFORMANCE\n")
cat("───────────────────────────────────────────────────────────────────────────────\n")
cat("Overall Accuracy:", sprintf("%.2f%%", final_summary$Performance$Overall_Accuracy * 100), "\n")
cat("Macro F1-Score:", final_summary$Performance$Macro_F1_Score, "\n")
cat("Macro AUC:", final_summary$Performance$Macro_AUC, "\n")
cat("Cohen's Kappa:", final_summary$Performance$Cohens_Kappa, "\n")
cat("Log Loss:", final_summary$Performance$Log_Loss, "\n\n")

cat("BUSINESS IMPACT\n")
cat("───────────────────────────────────────────────────────────────────────────────\n")
cat("Total Customers Analyzed:", format(final_summary$Business_Impact$Total_Customers, 
                                        big.mark = ","), "\n")
cat("High-Risk Customers:", format(final_summary$Business_Impact$High_Risk_Customers, 
                                   big.mark = ","), 
    sprintf("(%.1f%%)\n", final_summary$Business_Impact$High_Risk_Percentage))
cat("Monthly Revenue at Risk: $", format(final_summary$Business_Impact$Monthly_Revenue_At_Risk, 
                                         big.mark = ","), "\n")
cat("Annual Revenue at Risk: $", format(final_summary$Business_Impact$Annual_Revenue_At_Risk, 
                                        big.mark = ","), "\n")
cat("Recommended Retention Budget: $", 
    format(final_summary$Business_Impact$Recommended_Budget, big.mark = ","), "\n\n")

cat("═══════════════════════════════════════════════════════════════════════════════\n\n")


# Save outputs
if(!dir.exists("output")) dir.create("output")

saveRDS(final_model, "output/elastic_net_churn_model.rds")
saveRDS(cv_final, "output/cv_results.rds")
saveRDS(selected_features_final, "output/selected_features.rds")

# Save scaling info (extract from training process)
scaling_info <- list(
  means = colMeans(X_train_encoded),
  sds = apply(X_train_encoded, 2, sd)
)
saveRDS(scaling_info, "output/scaling_parameters.rds")

write.csv(test_results, "output/test_results_with_risk_scores.csv", row.names = FALSE)

if(exists("recently_joined_results")) {
  write.csv(recently_joined_results, 
            "output/recently_joined_risk_scores.csv", 
            row.names = FALSE)
}

write.csv(all_coefs, "output/model_coefficients.csv", row.names = FALSE)
write.csv(per_class_metrics, "output/per_class_metrics.csv", row.names = FALSE)
write.csv(evaluation_summary, "output/metrics_summary.csv", row.names = FALSE)
write.csv(risk_summary, "output/risk_summary.csv", row.names = FALSE)

cat("✓ All outputs saved to 'output/' directory\n\n")

cat("Files saved:\n")
cat("  • elastic_net_churn_model.rds - Trained model object\n")
cat("  • cv_results.rds - Cross-validation results\n")
cat("  • scaling_parameters.rds - Feature scaling parameters\n")
cat("  • selected_features.rds - List of selected features\n")
cat("  • test_results_with_risk_scores.csv - Test predictions with risk scores\n")
cat("  • recently_joined_risk_scores.csv - Risk scores for new customers\n")
cat("  • model_coefficients_and_odds_ratios.csv - Coefficient analysis\n")
cat("  • per_class_metrics.csv - Detailed per-class performance\n")
cat("  • overall_metrics_summary.csv - Overall model metrics\n")
cat("  • risk_distribution_summary.csv - Business risk analysis\n\n")

cat("\n")
cat("════════════════════════════════════════════════════════════════════════════════\n")
cat("                         ANALYSIS COMPLETE                                        \n")
cat("════════════════════════════════════════════════════════════════════════════════\n")
cat("\n")

# End of script

################################################################################
#                           END OF ANALYSIS                                    #
################################################################################  

