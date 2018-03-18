
loadCleanAmesData <- function() {
    
    zoning_df = read.csv('../data/zoning.csv')
    listing_df = read.csv('../data/listing.csv')
    sale_df = read.csv('../data/sale.csv')

    housing_df = merge(zoning_df, listing_df, by="Id")
    housing_df = merge(housing_df, sale_df, by="Id")

    # Remove Id column from data and set it to rownames
    rownames(housing_df) <- housing_df$Id
    #housing_df$Id <- NULL
    #head(housing_df, 4)

    # Convert certain numerical features that are actually categorical/Ordinal as factors
    housing_df$MSSubClass <- as.factor(housing_df$MSSubClass)
    housing_df$OverallQual <- as.factor(housing_df$OverallQual)
    housing_df$OverallCond <- as.factor(housing_df$OverallCond)
    housing_df$BsmtFullBath <- as.factor(housing_df$BsmtFullBath)
    housing_df$BsmtHalfBath <- as.factor(housing_df$BsmtHalfBath)
    housing_df$FullBath <- as.factor(housing_df$FullBath)
    housing_df$HalfBath <- as.factor(housing_df$HalfBath)
    housing_df$BedroomAbvGr <- as.factor(housing_df$BedroomAbvGr)
    housing_df$KitchenAbvGr <- as.factor(housing_df$KitchenAbvGr)
    housing_df$TotRmsAbvGrd <- as.factor(housing_df$TotRmsAbvGrd)
    housing_df$Fireplaces <- as.factor(housing_df$Fireplaces)
    housing_df$GarageCars <- as.factor(housing_df$GarageCars)
    housing_df$MoSold <- as.factor(housing_df$MoSold)
  

    # impute na values with mean 
    nan_sums = colSums(is.na(housing_df))
    nan_sums[nan_sums > 0]

    # Find the mean
    mean_LotFrontage <- mean(housing_df$LotFrontage, na.rm=T)
    mean_MasVnrArea <- mean(housing_df$MasVnrArea, na.rm=T)
    mean_GarageYrBlt <- mean(housing_df$GarageYrBlt, na.rm=T)

    # impute
    housing_df$LotFrontage[is.na(housing_df$LotFrontage)] <- mean_LotFrontage
    housing_df$MasVnrArea[is.na(housing_df$MasVnrArea)] <- mean_MasVnrArea
    housing_df$GarageYrBlt[is.na(housing_df$GarageYrBlt)] <- mean_GarageYrBlt

    # check for na values to make sure all na values have been imputed
    nan_sums = colSums(is.na(housing_df))
    nan_sums[nan_sums > 0]

    # impute blanks for categorical features
    levels(housing_df$Alley) <- c(levels(housing_df$Alley), "No alley access")
    housing_df$Alley[housing_df$Alley==""] <- "No alley access"

    levels(housing_df$BsmtQual) <- c(levels(housing_df$BsmtQual), "No Basement")
    housing_df$BsmtQual[housing_df$BsmtQual==""] <- "No Basement"

    levels(housing_df$BsmtCond) <- c(levels(housing_df$BsmtCond), "No Basement")
    housing_df$BsmtCond[housing_df$BsmtCond==""] <- "No Basement"

    levels(housing_df$BsmtExposure) <- c(levels(housing_df$BsmtExposure), "No Basement")
    housing_df$BsmtExposure[housing_df$BsmtExposure==""] <- "No Basement"

    levels(housing_df$BsmtFinType1) <- c(levels(housing_df$BsmtFinType1), "No Basement")
    housing_df$BsmtFinType1[housing_df$BsmtFinType1==""] <- "No Basement"

    levels(housing_df$BsmtFinType2) <- c(levels(housing_df$BsmtFinType2), "No Basement")
    housing_df$BsmtFinType2[housing_df$BsmtFinType2==""] <- "No Basement"

    levels(housing_df$FireplaceQu) <- c(levels(housing_df$FireplaceQu), "No Fireplace")
    housing_df$FireplaceQu[housing_df$FireplaceQu==""] <- "No Fireplace"

    levels(housing_df$GarageType) <- c(levels(housing_df$GarageType), "No Garage")
    housing_df$GarageType[housing_df$GarageType==""] <- "No Garage"

    levels(housing_df$GarageFinish) <- c(levels(housing_df$GarageFinish), "No Garage")
    housing_df$GarageFinish[housing_df$GarageFinish==""] <- "No Garage"

    levels(housing_df$GarageQual) <- c(levels(housing_df$GarageQual), "No Garage")
    housing_df$GarageQual[housing_df$GarageQual==""] <- "No Garage"

    levels(housing_df$GarageCond) <- c(levels(housing_df$GarageCond), "No Garage")
    housing_df$GarageCond[housing_df$GarageCond==""] <- "No Garage"

    levels(housing_df$PoolQC) <- c(levels(housing_df$PoolQC), "No Pool")
    housing_df$PoolQC[housing_df$PoolQC==""] <- "No Pool"

    levels(housing_df$Fence) <- c(levels(housing_df$Fence), "No Fence")
    housing_df$Fence[housing_df$Fence==""] <- "No Fence"

    levels(housing_df$MiscFeature) <- c(levels(housing_df$MiscFeature), "None")
    housing_df$MiscFeature[housing_df$MiscFeature==""] <- "None"
    
    levels(housing_df$Electrical) <- c(levels(housing_df$Electrical), "No Electrical")
    housing_df$Electrical[housing_df$Electrical==""] <- "No Electrical"
    
    levels(housing_df$MasVnrType) <- c(levels(housing_df$MasVnrType), "None")
    housing_df$MasVnrType[housing_df$MasVnrType==""] <- "None"
    
    return(housing_df)
}


count_blank <- function (dataframe, feature) {
    empty_string_mask = dataframe[feature] == ""
    return(length(dataframe[feature][empty_string_mask]))
}

colsums_is_blank <- function (dataframe){
    for (feature in colnames(dataframe)) {
        empty_count <- count_blank(dataframe, feature)
        if (empty_count > 0) {
            print(paste(feature, empty_count))        
        }
    }
}

replace_empty_with_without <- function(feature) {
    levels(feature) <- c(levels(feature), "without")
    empty_string_mask <- feature == ''
    feature[empty_string_mask] <- "without"
    return(feature)
}

replace_empty_with_NA <- function(feature) {
    levels(feature) <- c(levels(feature), NA)
    empty_string_mask <- feature == ''
    feature[empty_string_mask] <- NA
    return(feature)
}
