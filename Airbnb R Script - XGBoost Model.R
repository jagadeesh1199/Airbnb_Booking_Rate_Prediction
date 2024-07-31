library(tm)
library(SnowballC)
library(text2vec)
library(tidyverse)
library(caret)
library(tree)
library(class)
library(caret)
library(ggplot2)
library(ROCR)

#load data files
train_x <- read_csv("airbnb_train_x_2024.csv")
train_y <- read_csv("airbnb_train_y_2024.csv")
test_x <- read_csv("airbnb_test_x_2024.csv")

#---------------------------------------------------------------------

#Training data exploration

summary(train_x)

missing_data <- train_x[, colMeans(is.na(train_x)) > 0.4, drop = FALSE]
summary(missing_data)

no_unique <- train_x[, sapply(train_x, function(x) length(unique(x))) == 1, drop = FALSE]

missing_percentage <- colMeans(is.na(train_x)) * 100

# Identify columns with more than 50% missing values
columns_to_remove <- names(train_x)[missing_percentage > 50]

# Identify columns with only one unique value
unique_columns <- sapply(train_x, function(x) length(unique(x))) == 1
columns_to_remove <- c(columns_to_remove, names(unique_columns)[unique_columns])

summary(train_x[unique_columns])
# Remove identified columns from the dataset
cleaned_train_x <- train_x[, !(names(train_x) %in% columns_to_remove)]

# Print the cleaned dataset
summary(cleaned_train_x)

cleaned_train_x$high_booking_rate <- train_y$high_booking_rate

# Convert "high_booking_rate" to 0 and 1
cleaned_train_x$high_booking_rate <- ifelse(cleaned_train_x$high_booking_rate == "YES", 1, 0)
numeric_columns <- sapply(cleaned_train_x, is.numeric)
numeric_data <- cleaned_train_x[, numeric_columns]

# Drop rows with missing values
numeric_data <- na.omit(numeric_data)

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_data)
# Converting correlation matrix to dataframe
correlation_df <- as.data.frame(as.table(correlation_matrix))

# Plotting correlation matrix
ggplot(data = correlation_df, aes(x=Var1, y=Var2, fill=Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "lightskyblue3", mid = "lightblue2", high = "lightskyblue4", 
                       midpoint = 0, limits = c(-1,1), name="Correlation") +
  geom_text(aes(label = round(Freq, 2)), vjust = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Plot for Numeric Variables")

#Exploring some top features
#most top features are text suggesting that readers actually pay good attention
#to fine details in factors such as facilities and previous reviews 
# Count the number of columns with text data

text_columns <- sapply(train_x, function(x) is.character(x))
num_text_columns <- sum(text_columns)

# Print the number of columns with text data
print(num_text_columns)

#-----------------------------------------------------------------------
# TEXT MINING FOR EDA: 
#-----------------------------------------------------------------------
# Function to create TF-IDF features and ensure unique column names

# Create a text corpus
create_tfidf_features <- function(df, text_column, prefix) {
corpus <- Corpus(VectorSource(df[[text_column]]))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a document-term matrix with TF-IDF weighting
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))

dtm <- DocumentTermMatrix(corpus, control = list(
  weighting = weightTfIdf,
  wordLengths = c(2, Inf),  # Eliminate very short words
  bounds = list(global = c(15000, Inf))  # Terms must appear in at least 5 documents
))

dtm <- removeSparseTerms(dtm, 0.99)


tfidf <- as.matrix(dtm)
colnames(tfidf) <- make.names(colnames(tfidf), unique = TRUE)  # Ensure unique names

# Append a prefix to each column name based on the text column
colnames(tfidf) <- paste(prefix, colnames(tfidf), sep = "_")

library(Matrix)
tfidf_sparse <- as(tfidf, "sparseMatrix")

# Convert to data frame and return
tfidf_df <- as.data.frame(tfidf)
return(tfidf_df)
}


test_x_1 <- test_x

text_features <- rbind(train_x, test_x_1)


# Example usage with text columns
text_features$text_combined <- apply(text_features[c("description", "summary","neighborhood_overview","transit", "access","house_rules","city")], 1, paste, collapse=" ")
tfidf_features <- create_tfidf_features(text_features, "text_combined", "text")


# Assuming your dataframe is named 'df'
data_except_last_10000 <- tfidf_features[1:(nrow(tfidf_features)-10000), ]
last_10000_records <- tfidf_features[(nrow(tfidf_features)-9999):nrow(tfidf_features), ]

# Merge TF-IDF features into the original dataset
# First, check for duplicate column names that might arise
common_names <- intersect(names(train_x), names(data_except_last_10000))
if (length(common_names) > 0) {
  # Rename TF-IDF features to ensure no overlap
  names(data_except_last_10000) <- paste("tfidf", names(data_except_last_10000), sep = "_")
}

train_x <- cbind(train_x, data_except_last_10000)

train <- cbind(train_x, train_y) 

#---------------------------------------------------------------------
#EDA Graph Plots
#---------------------------------------------------------------------


hist(train$price, breaks = 20, col = "lightskyblue3",
     border = "black", main = "Histogram of Prices: 
     Most bookings are for cheaper rooms", 
     xlab = "Price", ylab = "Frequency") 


ggplot(train, aes(x = high_booking_rate, y = security_deposit, fill = high_booking_rate)) +
  geom_boxplot() +
  labs(x = "High Booking Rate", y = "Security Deposit", fill = "High Booking Rate") +
  ggtitle("Listing with lesser security deposit has higher booking rate") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


ggplot(train, aes(x = cleaning_fee, fill = high_booking_rate)) +
  geom_density(alpha = 0.5) + # Adjust transparency
  labs(x = "Availability_90", y = "Density") +
  ggtitle("Listings with low cleaning fee have highly booking rate") +
  scale_fill_manual(values = c("NO" = "blue", "YES" = "red")) + # Custom fill colors
  theme_minimal()+
  theme(
    plot.title = element_text(size = 11) 
  )


ggplot(train, aes(x = availability_90, fill = high_booking_rate)) +
  geom_density(alpha = 0.5) + # Adjust transparency
  labs(x = "Availability_90", y = "Density") +
  ggtitle("Listings with low booking rates have highly variablity in availability") +
  scale_fill_manual(values = c("NO" = "blue", "YES" = "red")) + # Custom fill colors
  theme_minimal()+
  theme(
    plot.title = element_text(size = 11) 
  )
#-------------------------------------------------------------------
# Random Forest to check which text features are most relevant
library(ranger)
train_y_2 <- factor(train_y$high_booking_rate, levels = c("YES", "NO"), labels = c(1, 0))
random_forest <- ranger(x = data_except_last_10000, y = train_y_2,
                        mtry = 30, num.trees = 300,
                        importance = "impurity", probability = TRUE)


library(vip)
vip(random_forest,num_features = 15) 

#-----------------------------------------------------------------------#-----------------------------------------------------------------------
#FINAL TRAIN DATASET CLEANING
#-------------------------------------------------------------------------------------

# Check for duplicate column names
print(anyDuplicated(names(train)))

# View duplicate column names
dup_cols <- names(train)[duplicated(names(train))]
print(dup_cols)

train <- train %>%
  mutate( high_booking_rate = as.factor(high_booking_rate),
          cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30"), "strict",    cancellation_policy),
          cleaning_fee = as.numeric(gsub("[^0-9.]", "", cleaning_fee)),
          price = as.numeric(gsub("[^0-9.]", "", price)),
          cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
          price = ifelse(is.na(price), 0, price),
          across(where(is.numeric) & !matches(c("cleaning_fee", "price")), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)),
          price_per_person = price/accommodates,
          has_cleaning_fee = as.factor(ifelse(as.numeric(cleaning_fee) ==0 | is.na(cleaning_fee) , "NO", "YES")),
          bed_category = ifelse(bed_type =="Real Bed", "bed", "other"),
          property_category = case_when(
            property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
            property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
            property_type %in% c("Townhouse", "Condominium") ~ "condo",
            property_type %in% c("Bungalow", "House") ~ "house",
            TRUE ~ "other"
          ),
          property_category = as.factor(property_category),
          bed_category = as.factor(bed_category),
          guests_included = as.factor(guests_included),
          market = ifelse(is.na(market), "Unknown", market),
          room_type = as.factor(room_type),
          state = as.factor(state)
  )

# Now try your mutation and grouping operations
train <- train %>%
  group_by(property_category) %>%
  mutate(median_ppp = median(price_per_person, na.rm = TRUE)) %>%
  ungroup()

# Continue with further transformations
train <- train %>%
  mutate(
    ppp_ind = ifelse(price_per_person > median_ppp, 1, 0),
    ppp_ind = factor(ppp_ind)
  )

train <- train %>%
  mutate(bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
         extra_people = as.numeric(gsub("[^0-9.]", "", extra_people)),
         charges_for_extra = as.factor(ifelse(is.na(extra_people) | extra_people == 0, "NO", "YES")),
         host_response_rate = as.numeric(gsub("[^0-9.]", "", host_response_rate)),
         host_acceptance_rate = as.numeric(gsub("[^0-9.]", "", host_acceptance_rate)),
         host_acceptance = as.factor(ifelse(is.na(host_acceptance_rate), "MISSING",
                                            ifelse(host_acceptance_rate == 100, "ALL", "SOME"))),
         host_response = as.factor(ifelse(is.na(host_response_rate), "MISSING",
                                          ifelse(host_response_rate == 100, "ALL", "SOME"))),
         has_min_nights = as.factor(ifelse(minimum_nights > 1, "YES", "NO"))
         
  ) %>%
  group_by(market) %>%
  mutate(market_instance = n()) %>%  # Count instances within each market
  ungroup() %>%
  mutate(market = as.factor(ifelse(market_instance < 300, "OTHER", market))
  )

summary(train)  

library(lubridate)

train <- train %>%
  mutate(
    host_since = as.Date(host_since, format = "%Y-%m-%d"),
    months_since_host_listed = as.numeric(interval(host_since, Sys.Date()) / months(1)),
    average_listings_per_month = ifelse(months_since_host_listed > 0, host_total_listings_count / months_since_host_listed, 0)
  )
mean_average_listings_per_month <- mean(train$average_listings_per_month, na.rm = TRUE)
train$average_listings_per_month <- ifelse(is.na(train$average_listings_per_month), mean_average_listings_per_month, train$average_listings_per_month)

mean_months_since_host_listed <- mean(train$months_since_host_listed, na.rm = TRUE)
train$months_since_host_listed <- ifelse(is.na(train$months_since_host_listed), mean_months_since_host_listed, train$months_since_host_listed)

train <- train %>%
  mutate(Free_parking_flag = factor(ifelse(grepl("Free parking", amenities), 1, 0)))

train <- train %>%
  mutate(Host_Is_Superhost_flag = factor(ifelse(grepl("Host Is Superhost", features), 1, 0)))

train <- train %>%
  mutate(G_verified_flag = factor(ifelse(grepl("google", host_verifications), 1, 0)))

data_train <- train %>% 
  mutate(interaction_term = accommodates * bedrooms,
         accommodates_s = accommodates^2,
  )

data_train <- train %>%
  mutate(
    superhost_response_interaction = interaction(Host_Is_Superhost_flag, host_response),
    cleaning_fee_price_interaction = ifelse(has_cleaning_fee == "YES", 1, 0) * price
  )

data_train <- data_train %>%
  mutate( workspace = as.factor(ifelse(grepl("workspace", amenities), 1, 0)),
         dryer = as.factor(ifelse(grepl("dryer", amenities), 1, 0)),
         board_game = as.factor(ifelse(grepl("board game", interaction), 1, 0)),
         home = as.factor(ifelse(grepl("home", interaction), 1, 0)),
         Free_parking_flag = as.factor(ifelse(grepl("Free parking", amenities), 1, 0)),
         wonderland = as.factor(ifelse(grepl("wonderland", neighborhood_overview), 1, 0))
         )

set.seed(1)  # for reproducibility
split_ratio <- 0.70  # 70% of data for training, 30% for testing
training_indices <- sample(nrow(data_train), split_ratio * nrow(data_train))
train_data <- data_train[training_indices, ]
test_data <- data_train[-training_indices, ]


train_data$high_booking_rate <- as.factor(train_data$high_booking_rate)
test_data$high_booking_rate <- as.factor(test_data$high_booking_rate)



library(caret)

# Define the control function with cross-validation
train_control <- trainControl(
  method = "cv",          # Cross-validation
  number = 2,            # Number of folds in the cross-validation
  savePredictions = "final",
  classProbs = TRUE,      # Save class probabilities (necessary for ROC metric)
  summaryFunction = twoClassSummary  # Use AUC as a performance metric
)

# Set up a custom tuning grid (optional)
xgb_grid <- expand.grid(
  nrounds = 600,  # Number of boosting rounds
  max_depth = 6, # Max depth of a tree
  eta = 0.1,     # Learning rate
  gamma = 0,      # Minimum loss reduction
  colsample_bytree = 0.75,  # Subsample ratio of columns
  min_child_weight =  3,        # Minimum sum of instance weight needed in a child
  subsample =  0.75         # Subsample ratio of the training instances
)


new_levels_test_data <- setdiff(levels(test_data$neighborhood), levels(test_x$neighborhood))
new_levels <- setdiff(new_levels_test_data, levels(train_data$neighborhood))

# Add new levels to the training data
train_data$neighborhood <- factor(train_data$neighborhood, levels = c(levels(train_data$neighborhood), new_levels))


# Identify columns with only one factor level
single_level_cols <- sapply(train_data, function(x) length(levels(factor(x))) == 1)

# Get the names of columns with only one factor level
single_level_col_names <- names(train_data)[single_level_cols]

# Check the identified columns
print(single_level_col_names)

train_data <- train_data[, !single_level_cols]

# Identify columns with NA values
na_cols <- colnames(train_data)[colSums(is.na(train_data)) > 0]

# Remove columns with NA values
train_data <- train_data[, !colnames(train_data) %in% na_cols]

train_data <- train_data %>%
  mutate(high_booking_rate = factor(high_booking_rate, levels = c("NO", "YES")))



levels(train_data$high_booking_rate)

columns_to_remove <- c("availability_30", "availability_60", "availability_365","smart_location","square_feet",
                       "maximum_nights","latitude","host_verifications","street","state","text_combined",
                       "first_review", "perfect_rating_score", "notes", "neighborhood_group",
                       "square_feet", "weekly_price", "monthly_price", "license", "jurisdiction_names")

# Use the select function with the - operator to exclude specified columns
train_data <- train_data %>%
  select(-one_of(columns_to_remove))



# Train the model using XGBoost

full_tree <- train(
  high_booking_rate ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  trControl = train_control,
 tuneGrid = xgb_grid  # Use the custom tuning grid
 )


summary(full_tree)

# Structuring test_data i.e validation
test_data$neighborhood <- factor(test_data$neighborhood, levels = c(levels(test_data$neighborhood), new_levels))


# Identify columns with only one factor level
single_level_cols <- sapply(test_data, function(x) length(levels(factor(x))) == 1)

# Get the names of columns with only one factor level
single_level_col_names <- names(test_data)[single_level_cols]

# Check the identified columns
print(single_level_col_names)

test_data <- test_data[, !single_level_cols]

# Identify columns with NA values
na_cols <- colnames(test_data)[colSums(is.na(test_data)) > 0]

# Remove columns with NA values
test_data <- test_data[, !colnames(test_data) %in% na_cols]

test_data <- test_data %>%
  mutate(high_booking_rate = factor(high_booking_rate, levels = c("NO", "YES")))

levels(test_data$high_booking_rate)

columns_to_remove <- c("availability_30", "availability_60", "availability_365","smart_location","square_feet",
                       "maximum_nights","latitude","host_verifications","street","state","text_combined",
                       "first_review", "perfect_rating_score", "notes", "neighborhood_group",
                       "square_feet", "weekly_price", "monthly_price", "license", "jurisdiction_names")

# Use the select function with the - operator to exclude specified columns
test_data <- test_data %>%
  select(-one_of(columns_to_remove))


#-----------------------------------------------------------#
# Model Evaluation using the testing set
#-----------------------------------------------------------#
predictions <- predict(full_tree, test_data, type = "prob")[,2]

print(class(predictions)) 

library(pROC)
roc_curve <- roc(test_data$high_booking_rate, predictions)

# Plot the ROC curve and print the AUC
plot(roc_curve, main = "ROC Curve")
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))


predictions <- predict(full_tree, newdata = test_data)
conf_matrix <- confusionMatrix(predictions, test_data$high_booking_rate)

# Display the confusion matrix and accuracy
print(conf_matrix)
print(conf_matrix$overall["Accuracy"])

# Access the table
conf_matrix$table

# Access the overall accuracy
conf_matrix$overall["Accuracy"]
TP <- conf_matrix$table[2,2]
FN <- conf_matrix$table[2,1]
FP <- conf_matrix$table[1,2]
TN <- conf_matrix$table[1,1]

# Calculate
TPR <- TP / (TP + FN)
FPR <- FP / (FP + TN)

# Print results
print(paste("TPR:", TPR))
print(paste("FPR:", FPR))

#------------------------------------------------------------------------#
# structuring test_x to make predictions
#------------------------------------------------------------------------#


common_names <- intersect(names(test_x), names(last_10000_records))
if (length(common_names) > 0) {
  # Rename TF-IDF features to ensure no overlap
  names(last_10000_records) <- paste("tfidf", names(last_10000_records), sep = "_")
}

test_x <- cbind(test_x, last_10000_records)

test_x <- test_x %>%
  mutate( 
          cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30"), "strict",    cancellation_policy),
          cleaning_fee = as.numeric(gsub("[^0-9.]", "", cleaning_fee)),
          price = as.numeric(gsub("[^0-9.]", "", price)),
          cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
          price = ifelse(is.na(price), 0, price),
          across(where(is.numeric) & !matches(c("cleaning_fee", "price")), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)),
          price_per_person = price/accommodates,
          has_cleaning_fee = as.factor(ifelse(as.numeric(cleaning_fee) ==0 | is.na(cleaning_fee) , "NO", "YES")),
          bed_category = ifelse(bed_type =="Real Bed", "bed", "other"),
          property_category = case_when(
            property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
            property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
            property_type %in% c("Townhouse", "Condominium") ~ "condo",
            property_type %in% c("Bungalow", "House") ~ "house",
            TRUE ~ "other"
          ),
          property_category = as.factor(property_category),
          bed_category = as.factor(bed_category),
          guests_included = as.factor(guests_included),
          market = ifelse(is.na(market), "Unknown", market),
          room_type = as.factor(room_type),
          state = as.factor(state)
  )


test_x <- test_x %>%
  group_by(property_category) %>%
  mutate(median_ppp = median(price_per_person, na.rm = TRUE)) %>%
  ungroup()

# Continue with further transformations
test_x <- test_x %>%
  mutate(
    ppp_ind = ifelse(price_per_person > median_ppp, 1, 0),
    ppp_ind = factor(ppp_ind)
  )

test_x <- test_x %>%
  mutate(bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
         extra_people = as.numeric(gsub("[^0-9.]", "", extra_people)),
         charges_for_extra = as.factor(ifelse(is.na(extra_people) | extra_people == 0, "NO", "YES")),
         host_response_rate = as.numeric(gsub("[^0-9.]", "", host_response_rate)),
         host_acceptance_rate = as.numeric(gsub("[^0-9.]", "", host_acceptance_rate)),
         host_acceptance = as.factor(ifelse(is.na(host_acceptance_rate), "MISSING",
                                            ifelse(host_acceptance_rate == 100, "ALL", "SOME"))),
         host_response = as.factor(ifelse(is.na(host_response_rate), "MISSING",
                                          ifelse(host_response_rate == 100, "ALL", "SOME"))),
         has_min_nights = as.factor(ifelse(minimum_nights > 1, "YES", "NO"))
         
  ) %>%
  group_by(market) %>%
  mutate(market_instance = n()) %>%  # Count instances within each market
  ungroup() %>%
  mutate(market = as.factor(ifelse(market_instance < 300, "OTHER", market))
  )

summary(test_x)  

library(lubridate)

test_x <- test_x %>%
  mutate(
    host_since = as.Date(host_since, format = "%Y-%m-%d"),
    months_since_host_listed = as.numeric(interval(host_since, Sys.Date()) / months(1)),
    average_listings_per_month = ifelse(months_since_host_listed > 0, host_total_listings_count / months_since_host_listed, 0)
  )
mean_average_listings_per_month <- mean(test_x$average_listings_per_month, na.rm = TRUE)
test_x$average_listings_per_month <- ifelse(is.na(test_x$average_listings_per_month), mean_average_listings_per_month, train$average_listings_per_month)

mean_months_since_host_listed <- mean(test_x$months_since_host_listed, na.rm = TRUE)
test_x$months_since_host_listed <- ifelse(is.na(test_x$months_since_host_listed), mean_months_since_host_listed, test_x$months_since_host_listed)

test_x <- test_x %>%
  mutate(Free_parking_flag = factor(ifelse(grepl("Free parking", amenities), 1, 0)))

test_x <- test_x %>%
  mutate(Host_Is_Superhost_flag = factor(ifelse(grepl("Host Is Superhost", features), 1, 0)))

test_x <- test_x %>%
  mutate(G_verified_flag = factor(ifelse(grepl("google", host_verifications), 1, 0)))

test_x <- test_x %>% 
  mutate(interaction_term = accommodates * bedrooms,
         accommodates_s = accommodates^2,
  )

test_x <- test_x %>%
  mutate( workspace = as.factor(ifelse(grepl("workspace", amenities), 1, 0)),
          dryer = as.factor(ifelse(grepl("dryer", amenities), 1, 0)),
          board_game = as.factor(ifelse(grepl("board game", interaction), 1, 0)),
          home = as.factor(ifelse(grepl("home", interaction), 1, 0)),
          Free_parking_flag = as.factor(ifelse(grepl("Free parking", amenities), 1, 0)),
          wonderland = as.factor(ifelse(grepl("wonderland", neighborhood_overview), 1, 0))
  )

test_x <- test_x %>%
  mutate(
    superhost_response_interaction = interaction(Host_Is_Superhost_flag, host_response),
    cleaning_fee_price_interaction = ifelse(has_cleaning_fee == "YES", 1, 0) * price
  )



# Add new levels to the test data
test_x$neighborhood <- factor(test_x$neighborhood, levels = c(levels(train_data$neighborhood), new_levels))


# Identify columns with only one factor level
single_level_cols <- sapply(test_x, function(x) length(levels(factor(x))) == 1)

# Get the names of columns with only one factor level
single_level_col_names <- names(test_x)[single_level_cols]

# Check the identified columns
print(single_level_col_names)

test_x <- test_x[, !single_level_cols]

# Identify columns with NA values
na_cols <- colnames(test_x)[colSums(is.na(test_x)) > 0]

# Remove columns with NA values
test_x <- test_x[, !colnames(test_x) %in% na_cols]


columns_to_remove <- c("availability_30", "availability_60", "availability_365","smart_location","square_feet",
                       "maximum_nights","latitude","host_verifications","street","state","text_combined",
                       "first_review", "perfect_rating_score", "notes", "neighborhood_group",
                       "square_feet", "weekly_price", "monthly_price", "license", "jurisdiction_names")

# Use the select function with the - operator to exclude specified columns
test_x <- test_x %>%
  select(-one_of(columns_to_remove))

# Identify columns with only one factor level
single_level_cols <- sapply(test_x, function(x) length(levels(factor(x))) == 1)

# Get the names of columns with only one factor level
single_level_col_names <- names(test_x)[single_level_cols]

# Check the identified columns
print(single_level_col_names)

test_x <- test_x[, !single_level_cols]

# Identify columns with NA values
na_cols <- colnames(test_x)[colSums(is.na(test_x)) > 0]

# Remove columns with NA values
test_x <- test_x[, !colnames(test_x) %in% na_cols]


# make the prediction on the test_x dataset
predictions <- predict(full_tree, test_x, type = "prob")[,2]

# create excel file with prediction for text_x
write.table(predictions, "high_booking_rate_group1.csv", row.names = FALSE)

  #--------------------------------------------------------------------------------#

