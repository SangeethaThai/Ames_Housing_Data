# CSX_450_2_Final_Exam


### Domain

Data set contains information from the Ames Assessor’s Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010.

The type of information contained in the data is similar to what a typical home buyer would want to know before making a purchase.



### Problem Statement

This is a Regression problem in which the challenge is to predict the Sale Price of residential properties sold in Ames, IOWA given the house features such as neighborhood, square footage, number of full bathrooms, etc. 

The task $T$ is a Regression and the experience $E$ is the list of houses and their sale price.
**Task (T):** Use Regression to predict the Sale Price of houses in Ames, Iowa

**Experience (E):** A Corpus of Housing Data containing information you would want to know before purchasing a residential property in Ames, Iowa.

**Performance (P):** The accuracy with which the Sale Price of a residential property in Ames, Iowa is predicted as a percentage of the total number of houses for which Sale Price is predicted.



### Data Description

The data has 1460 observations and 81 columns which include 1 Id column which identifies the Observation number, 79 features (22 nominal, 23 ordinal, 14 discrete, and 20 continuous variables) and a Target (continuous) variable Sales Price.

##### Size of the data when loaded in memory is 608 KB

    Id (Discrete): Observation number
    
   #### Features     
    
    MS SubClass (Nominal): Identifies the type of dwelling involved in the sale.	

           020	1-STORY 1946 & NEWER ALL STYLES
           030	1-STORY 1945 & OLDER
           040	1-STORY W/FINISHED ATTIC ALL AGES
           045	1-1/2 STORY - UNFINISHED ALL AGES
           050	1-1/2 STORY FINISHED ALL AGES
           060	2-STORY 1946 & NEWER
           070	2-STORY 1945 & OLDER
           075	2-1/2 STORY ALL AGES
           080	SPLIT OR MULTI-LEVEL
           085	SPLIT FOYER
           090	DUPLEX - ALL STYLES AND AGES
           120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
           150	1-1/2 STORY PUD - ALL AGES
           160	2-STORY PUD - 1946 & NEWER
           180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
           190	2 FAMILY CONVERSION - ALL STYLES AND AGES

    MS Zoning (Nominal): Identifies the general zoning classification of the sale.

           A	Agriculture
           C	Commercial
           FV	Floating Village Residential
           I	Industrial
           RH	Residential High Density
           RL	Residential Low Density
           RP	Residential Low Density Park 
           RM	Residential Medium Density

    Lot Frontage (Continuous): Linear feet of street connected to property

    Lot Area (Continuous): Lot size in square feet

    Street (Nominal): Type of road access to property

           Grvl	Gravel	
           Pave	Paved

    Alley (Nominal): Type of alley access to property

           Grvl	Gravel
           Pave	Paved
           NA 	No alley access

    Lot Shape (Ordinal): General shape of property

           Reg	Regular	
           IR1	Slightly irregular
           IR2	Moderately Irregular
           IR3	Irregular

    Land Contour (Nominal): Flatness of the property

           Lvl	Near Flat/Level	
           Bnk	Banked - Quick and significant rise from street grade to building
           HLS	Hillside - Significant slope from side to side
           Low	Depression

    Utilities (Ordinal): Type of utilities available

           AllPub	All public Utilities (E,G,W,& S)	
           NoSewr	Electricity, Gas, and Water (Septic Tank)
           NoSeWa	Electricity and Gas Only
           ELO	Electricity only	

    Lot Config (Nominal): Lot configuration

           Inside	Inside lot
           Corner	Corner lot
           CulDSac	Cul-de-sac
           FR2	Frontage on 2 sides of property
           FR3	Frontage on 3 sides of property

    Land Slope (Ordinal): Slope of property

           Gtl	Gentle slope
           Mod	Moderate Slope	
           Sev	Severe Slope

    Neighborhood (Nominal): Physical locations within Ames city limits (map available)

           Blmngtn	Bloomington Heights
           Blueste	Bluestem
           BrDale	Briardale
           BrkSide	Brookside
           ClearCr	Clear Creek
           CollgCr	College Creek
           Crawfor	Crawford
           Edwards	Edwards
           Gilbert	Gilbert
           Greens	Greens
           GrnHill	Green Hills
           IDOTRR	Iowa DOT and Rail Road
           Landmrk	Landmark
           MeadowV	Meadow Village
           Mitchel	Mitchell
           Names	North Ames
           NoRidge	Northridge
           NPkVill	Northpark Villa
           NridgHt	Northridge Heights
           NWAmes	Northwest Ames
           OldTown	Old Town
           SWISU	South & West of Iowa State University
           Sawyer	Sawyer
           SawyerW	Sawyer West
           Somerst	Somerset
           StoneBr	Stone Brook
           Timber	Timberland
           Veenker	Veenker

    Condition 1 (Nominal): Proximity to various conditions

           Artery	Adjacent to arterial street
           Feedr	Adjacent to feeder street	
           Norm	Normal	
           RRNn	Within 200' of North-South Railroad
           RRAn	Adjacent to North-South Railroad
           PosN	Near positive off-site feature--park, greenbelt, etc.
           PosA	Adjacent to postive off-site feature
           RRNe	Within 200' of East-West Railroad
           RRAe	Adjacent to East-West Railroad

    Condition 2 (Nominal): Proximity to various conditions (if more than one is present)

           Artery	Adjacent to arterial street
           Feedr	Adjacent to feeder street	
           Norm	Normal	
           RRNn	Within 200' of North-South Railroad
           RRAn	Adjacent to North-South Railroad
           PosN	Near positive off-site feature--park, greenbelt, etc.
           PosA	Adjacent to postive off-site feature
           RRNe	Within 200' of East-West Railroad
           RRAe	Adjacent to East-West Railroad

    Bldg Type (Nominal): Type of dwelling

           1Fam	Single-family Detached	
           2FmCon	Two-family Conversion; originally built as one-family dwelling
           Duplx	Duplex
           TwnhsE	Townhouse End Unit
           TwnhsI	Townhouse Inside Unit

    House Style (Nominal): Style of dwelling

           1Story	One story
           1.5Fin	One and one-half story: 2nd level finished
           1.5Unf	One and one-half story: 2nd level unfinished
           2Story	Two story
           2.5Fin	Two and one-half story: 2nd level finished
           2.5Unf	Two and one-half story: 2nd level unfinished
           SFoyer	Split Foyer
           SLvl	Split Level

    Overall Qual (Ordinal): Rates the overall material and finish of the house

           10	Very Excellent
           9	Excellent
           8	Very Good
           7	Good
           6	Above Average
           5	Average
           4	Below Average
           3	Fair
           2	Poor
           1	Very Poor

    Overall Cond (Ordinal): Rates the overall condition of the house

           10	Very Excellent
           9	Excellent
           8	Very Good
           7	Good
           6	Above Average	
           5	Average
           4	Below Average	
           3	Fair
           2	Poor
           1	Very Poor

    Year Built (Discrete): Original construction date

    Year Remod/Add (Discrete): Remodel date (same as construction date if no remodeling or additions)

    Roof Style (Nominal): Type of roof

           Flat	Flat
           Gable	Gable
           Gambrel	Gabrel (Barn)
           Hip	Hip
           Mansard	Mansard
           Shed	Shed

    Roof Matl (Nominal): Roof material

           ClyTile	Clay or Tile
           CompShg	Standard (Composite) Shingle
           Membran	Membrane
           Metal	Metal
           Roll	Roll
           Tar&Grv	Gravel & Tar
           WdShake	Wood Shakes
           WdShngl	Wood Shingles

    Exterior 1 (Nominal): Exterior covering on house

           AsbShng	Asbestos Shingles
           AsphShn	Asphalt Shingles
           BrkComm	Brick Common
           BrkFace	Brick Face
           CBlock	Cinder Block
           CemntBd	Cement Board
           HdBoard	Hard Board
           ImStucc	Imitation Stucco
           MetalSd	Metal Siding
           Other	Other
           Plywood	Plywood
           PreCast	PreCast	
           Stone	Stone
           Stucco	Stucco
           VinylSd	Vinyl Siding
           Wd Sdng	Wood Siding
           WdShing	Wood Shingles

    Exterior 2 (Nominal): Exterior covering on house (if more than one material)

           AsbShng	Asbestos Shingles
           AsphShn	Asphalt Shingles
           BrkComm	Brick Common
           BrkFace	Brick Face
           CBlock	Cinder Block
           CemntBd	Cement Board
           HdBoard	Hard Board
           ImStucc	Imitation Stucco
           MetalSd	Metal Siding
           Other	Other
           Plywood	Plywood
           PreCast	PreCast
           Stone	Stone
           Stucco	Stucco
           VinylSd	Vinyl Siding
           Wd Sdng	Wood Siding
           WdShing	Wood Shingles

    Mas Vnr Type (Nominal): Masonry veneer type

           BrkCmn	Brick Common
           BrkFace	Brick Face
           CBlock	Cinder Block
           None	None
           Stone	Stone

    Mas Vnr Area (Continuous): Masonry veneer area in square feet

    Exter Qual (Ordinal): Evaluates the quality of the material on the exterior 

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           Po	Poor

    Exter Cond (Ordinal): Evaluates the present condition of the material on the exterior

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           Po	Poor

    Foundation (Nominal): Type of foundation

           BrkTil	Brick & Tile
           CBlock	Cinder Block
           PConc	Poured Contrete	
           Slab	Slab
           Stone	Stone
           Wood	Wood

    Bsmt Qual (Ordinal): Evaluates the height of the basement

           Ex	Excellent (100+ inches)	
           Gd	Good (90-99 inches)
           TA	Typical (80-89 inches)
           Fa	Fair (70-79 inches)
           Po	Poor (<70 inches
           NA	No Basement

    Bsmt Cond (Ordinal): Evaluates the general condition of the basement

           Ex	Excellent
           Gd	Good
           TA	Typical - slight dampness allowed
           Fa	Fair - dampness or some cracking or settling
           Po	Poor - Severe cracking, settling, or wetness
           NA	No Basement

    Bsmt Exposure	(Ordinal): Refers to walkout or garden level walls

           Gd	Good Exposure
           Av	Average Exposure (split levels or foyers typically score average or above)	
           Mn	Mimimum Exposure
           No	No Exposure
           NA	No Basement

    BsmtFin Type 1	(Ordinal): Rating of basement finished area

           GLQ	Good Living Quarters
           ALQ	Average Living Quarters
           BLQ	Below Average Living Quarters	
           Rec	Average Rec Room
           LwQ	Low Quality
           Unf	Unfinshed
           NA	No Basement

    BsmtFin SF 1 (Continuous): Type 1 finished square feet

    BsmtFinType 2	(Ordinal): Rating of basement finished area (if multiple types)

           GLQ	Good Living Quarters
           ALQ	Average Living Quarters
           BLQ	Below Average Living Quarters	
           Rec	Average Rec Room
           LwQ	Low Quality
           Unf	Unfinshed
           NA	No Basement

    BsmtFin SF 2 (Continuous): Type 2 finished square feet

    Bsmt Unf SF (Continuous): Unfinished square feet of basement area

    Total Bsmt SF (Continuous): Total square feet of basement area

    Heating	(Nominal): Type of heating

           Floor	Floor Furnace
           GasA	Gas forced warm air furnace
           GasW	Gas hot water or steam heat
           Grav	Gravity furnace	
           OthW	Hot water or steam heat other than gas
           Wall	Wall furnace

    HeatingQC (Ordinal): Heating quality and condition

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           Po	Poor

    Central Air (Nominal): Central air conditioning

           N	No
           Y	Yes

    Electrical (Ordinal): Electrical system

           SBrkr	Standard Circuit Breakers & Romex
           FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
           FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
           FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
           Mix	Mixed

    1st Flr SF (Continuous): First Floor square feet

    2nd Flr SF (Continuous)	: Second floor square feet

    Low Qual Fin SF (Continuous): Low quality finished square feet (all floors)

    Gr Liv Area (Continuous): Above grade (ground) living area square feet

    Bsmt Full Bath (Discrete): Basement full bathrooms

    Bsmt Half Bath (Discrete): Basement half bathrooms

    Full Bath (Discrete): Full bathrooms above grade

    Half Bath (Discrete): Half baths above grade

    Bedroom (Discrete): Bedrooms above grade (does NOT include basement bedrooms)

    Kitchen (Discrete): Kitchens above grade

    KitchenQual (Ordinal): Kitchen quality

           Ex	Excellent
           Gd	Good
           TA	Typical/Average
           Fa	Fair
           Po	Poor

    TotRmsAbvGrd	(Discrete): Total rooms above grade (does not include bathrooms)

    Functional (Ordinal): Home functionality (Assume typical unless deductions are warranted)

           Typ	Typical Functionality
           Min1	Minor Deductions 1
           Min2	Minor Deductions 2
           Mod	Moderate Deductions
           Maj1	Major Deductions 1
           Maj2	Major Deductions 2
           Sev	Severely Damaged
           Sal	Salvage only

    Fireplaces (Discrete): Number of fireplaces

    FireplaceQu (Ordinal): Fireplace quality

           Ex	Excellent - Exceptional Masonry Fireplace
           Gd	Good - Masonry Fireplace in main level
           TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
           Fa	Fair - Prefabricated Fireplace in basement
           Po	Poor - Ben Franklin Stove
           NA	No Fireplace

    Garage Type (Nominal): Garage location

           2Types	More than one type of garage
           Attchd	Attached to home
           Basment	Basement Garage
           BuiltIn	Built-In (Garage part of house - typically has room above garage)
           CarPort	Car Port
           Detchd	Detached from home
           NA	No Garage

    Garage Yr Blt (Discrete): Year garage was built

    Garage Finish (Ordinal)	: Interior finish of the garage

           Fin	Finished
           RFn	Rough Finished	
           Unf	Unfinished
           NA	No Garage

    Garage Cars (Discrete): Size of garage in car capacity

    Garage Area (Continuous): Size of garage in square feet

    Garage Qual (Ordinal): Garage quality

           Ex	Excellent
           Gd	Good
           TA	Typical/Average
           Fa	Fair
           Po	Poor
           NA	No Garage

    Garage Cond (Ordinal): Garage condition

           Ex	Excellent
           Gd	Good
           TA	Typical/Average
           Fa	Fair
           Po	Poor
           NA	No Garage

    Paved Drive (Ordinal): Paved driveway

           Y	Paved 
           P	Partial Pavement
           N	Dirt/Gravel

    Wood Deck SF (Continuous): Wood deck area in square feet

    Open Porch SF (Continuous): Open porch area in square feet

    Enclosed Porch (Continuous): Enclosed porch area in square feet

    3-Ssn Porch (Continuous): Three season porch area in square feet

    Screen Porch (Continuous): Screen porch area in square feet

    Pool Area (Continuous): Pool area in square feet

    Pool QC (Ordinal): Pool quality

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           NA	No Pool

    Fence (Ordinal): Fence quality

           GdPrv	Good Privacy
           MnPrv	Minimum Privacy
           GdWo	Good Wood
           MnWw	Minimum Wood/Wire
           NA	No Fence

    Misc Feature (Nominal): Miscellaneous feature not covered in other categories

           Elev	Elevator
           Gar2	2nd Garage (if not described in garage section)
           Othr	Other
           Shed	Shed (over 100 SF)
           TenC	Tennis Court
           NA	None

    Misc Val (Continuous): $Value of miscellaneous feature

    Mo Sold (Discrete): Month Sold (MM)

    Yr Sold (Discrete): Year Sold (YYYY)

    Sale Type (Nominal): Type of sale

           WD 	Warranty Deed - Conventional
           CWD	Warranty Deed - Cash
           VWD	Warranty Deed - VA Loan
           New	Home just constructed and sold
           COD	Court Officer Deed/Estate
           Con	Contract 15% Down payment regular terms
           ConLw	Contract Low Down payment and low interest
           ConLI	Contract Low Interest
           ConLD	Contract Low Down
           Oth	Other

    Sale Condition (Nominal): Condition of sale

           Normal	Normal Sale
           Abnorml	Abnormal Sale -  trade, foreclosure, short sale
           AdjLand	Adjoining Land Purchase
           Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
           Family	Sale between family members
           Partial	Home was not completed when last assessed (associated with New Homes)


   #### Target
    
    SalePrice (Continuous): Sale price $$ 



#### Summaries
```
   MSSubClass       MSZoning     LotFrontage        LotArea       LotShape 
 Min.   : 20.0   C (all):  10   Min.   : 21.00   Min.   :  1300   IR1:484  
 1st Qu.: 20.0   FV     :  65   1st Qu.: 60.00   1st Qu.:  7554   IR2: 41  
 Median : 50.0   RH     :  16   Median : 70.05   Median :  9478   IR3: 10  
 Mean   : 56.9   RL     :1151   Mean   : 70.05   Mean   : 10517   Reg:925  
 3rd Qu.: 70.0   RM     : 218   3rd Qu.: 79.00   3rd Qu.: 11602            
 Max.   :190.0                  Max.   :313.00   Max.   :215245            
                                                                           
 LandContour  Utilities      LotConfig    LandSlope   Neighborhood
 Bnk:  63    AllPub:1459   Corner : 263   Gtl:1382   NAmes  :225  
 HLS:  50    NoSeWa:   1   CulDSac:  94   Mod:  65   CollgCr:150  
 Low:  36                  FR2    :  47   Sev:  13   OldTown:113  
 Lvl:1311                  FR3    :   4              Edwards:100  
                           Inside :1052              Somerst: 86  
                                                     Gilbert: 79  
                                                     (Other):707  
   Condition1     Condition2     BldgType      HouseStyle   Street    
 Norm   :1260   Norm   :1445   1Fam  :1220   1Story :726   Grvl:   6  
 Feedr  :  81   Feedr  :   6   2fmCon:  31   2Story :445   Pave:1454  
 Artery :  48   Artery :   2   Duplex:  52   1.5Fin :154              
 RRAn   :  26   PosN   :   2   Twnhs :  43   SLvl   : 65              
 PosN   :  19   RRNn   :   2   TwnhsE: 114   SFoyer : 37              
 RRAe   :  11   PosA   :   1                 1.5Unf : 14              
 (Other):  15   (Other):   2                 (Other): 19              
    Alley            OverallQual      OverallCond      YearBuilt   
 Length:1460        Min.   : 1.000   Min.   :1.000   Min.   :1872  
 Class :character   1st Qu.: 5.000   1st Qu.:5.000   1st Qu.:1954  
 Mode  :character   Median : 6.000   Median :5.000   Median :1973  
                    Mean   : 6.099   Mean   :5.575   Mean   :1971  
                    3rd Qu.: 7.000   3rd Qu.:6.000   3rd Qu.:2000  
                    Max.   :10.000   Max.   :9.000   Max.   :2010  
                                                                   
  YearRemodAdd    RoofStyle       RoofMatl     Exterior1st   Exterior2nd 
 Min.   :1950   Flat   :  13   CompShg:1434   VinylSd:515   VinylSd:504  
 1st Qu.:1967   Gable  :1141   Tar&Grv:  11   HdBoard:222   MetalSd:214  
 Median :1994   Gambrel:  11   WdShngl:   6   MetalSd:220   HdBoard:207  
 Mean   :1985   Hip    : 286   WdShake:   5   Wd Sdng:206   Wd Sdng:197  
 3rd Qu.:2004   Mansard:   7   ClyTile:   1   Plywood:108   Plywood:142  
 Max.   :2010   Shed   :   2   Membran:   1   CemntBd: 61   CmentBd: 60  
                               (Other):   2   (Other):128   (Other):136  
  MasVnrType          MasVnrArea     ExterQual ExterCond  Foundation 
 Length:1460        Min.   :   0.0   Ex: 52    Ex:   3   BrkTil:146  
 Class :character   1st Qu.:   0.0   Fa: 14    Fa:  28   CBlock:634  
 Mode  :character   Median :   0.0   Gd:488    Gd: 146   PConc :647  
                    Mean   : 103.7   TA:906    Po:   1   Slab  : 24  
                    3rd Qu.: 164.2             TA:1282   Stone :  6  
                    Max.   :1600.0                       Wood  :  3  
                                                                     
   BsmtQual           BsmtCond         BsmtExposure       BsmtFinType1      
 Length:1460        Length:1460        Length:1460        Length:1460       
 Class :character   Class :character   Class :character   Class :character  
 Mode  :character   Mode  :character   Mode  :character   Mode  :character  
                                                                            
                                                                            
                                                                            
                                                                            
   BsmtFinSF1     BsmtFinType2         BsmtFinSF2        BsmtUnfSF     
 Min.   :   0.0   Length:1460        Min.   :   0.00   Min.   :   0.0  
 1st Qu.:   0.0   Class :character   1st Qu.:   0.00   1st Qu.: 223.0  
 Median : 383.5   Mode  :character   Median :   0.00   Median : 477.5  
 Mean   : 443.6                      Mean   :  46.55   Mean   : 567.2  
 3rd Qu.: 712.2                      3rd Qu.:   0.00   3rd Qu.: 808.0  
 Max.   :5644.0                      Max.   :1474.00   Max.   :2336.0  
                                                                       
  TotalBsmtSF      Heating     HeatingQC CentralAir  Electrical       
 Min.   :   0.0   Floor:   1   Ex:741    N:  95     Length:1460       
 1st Qu.: 795.8   GasA :1428   Fa: 49    Y:1365     Class :character  
 Median : 991.5   GasW :  18   Gd:241               Mode  :character  
 Mean   :1057.4   Grav :   7   Po:  1                                 
 3rd Qu.:1298.2   OthW :   2   TA:428                                 
 Max.   :6110.0   Wall :   4                                          
                                                                      
   FirstFlrSF    SecondFlrSF    LowQualFinSF       GrLivArea   
 Min.   : 334   Min.   :   0   Min.   :  0.000   Min.   : 334  
 1st Qu.: 882   1st Qu.:   0   1st Qu.:  0.000   1st Qu.:1130  
 Median :1087   Median :   0   Median :  0.000   Median :1464  
 Mean   :1163   Mean   : 347   Mean   :  5.845   Mean   :1515  
 3rd Qu.:1391   3rd Qu.: 728   3rd Qu.:  0.000   3rd Qu.:1777  
 Max.   :4692   Max.   :2065   Max.   :572.000   Max.   :5642  
                                                               
  BsmtFullBath     BsmtHalfBath        FullBath        HalfBath     
 Min.   :0.0000   Min.   :0.00000   Min.   :0.000   Min.   :0.0000  
 1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:1.000   1st Qu.:0.0000  
 Median :0.0000   Median :0.00000   Median :2.000   Median :0.0000  
 Mean   :0.4253   Mean   :0.05753   Mean   :1.565   Mean   :0.3829  
 3rd Qu.:1.0000   3rd Qu.:0.00000   3rd Qu.:2.000   3rd Qu.:1.0000  
 Max.   :3.0000   Max.   :2.00000   Max.   :3.000   Max.   :2.0000  
                                                                    
  BedroomAbvGr    KitchenAbvGr   KitchenQual  TotRmsAbvGrd    Functional 
 Min.   :0.000   Min.   :0.000   Ex:100      Min.   : 2.000   Maj1:  14  
 1st Qu.:2.000   1st Qu.:1.000   Fa: 39      1st Qu.: 5.000   Maj2:   5  
 Median :3.000   Median :1.000   Gd:586      Median : 6.000   Min1:  31  
 Mean   :2.866   Mean   :1.047   TA:735      Mean   : 6.518   Min2:  34  
 3rd Qu.:3.000   3rd Qu.:1.000               3rd Qu.: 7.000   Mod :  15  
 Max.   :8.000   Max.   :3.000               Max.   :14.000   Sev :   1  
                                                              Typ :1360  
   Fireplaces    FireplaceQu         GarageType         GarageYrBlt  
 Min.   :0.000   Length:1460        Length:1460        Min.   :1900  
 1st Qu.:0.000   Class :character   Class :character   1st Qu.:1962  
 Median :1.000   Mode  :character   Mode  :character   Median :1979  
 Mean   :0.613                                         Mean   :1979  
 3rd Qu.:1.000                                         3rd Qu.:2001  
 Max.   :3.000                                         Max.   :2010  
                                                                     
 GarageFinish         GarageCars      GarageArea      GarageQual       
 Length:1460        Min.   :0.000   Min.   :   0.0   Length:1460       
 Class :character   1st Qu.:1.000   1st Qu.: 334.5   Class :character  
 Mode  :character   Median :2.000   Median : 480.0   Mode  :character  
                    Mean   :1.767   Mean   : 473.0                     
                    3rd Qu.:2.000   3rd Qu.: 576.0                     
                    Max.   :4.000   Max.   :1418.0                     
                                                                       
  GarageCond        PavedDrive   WoodDeckSF      OpenPorchSF    
 Length:1460        N:  90     Min.   :  0.00   Min.   :  0.00  
 Class :character   P:  30     1st Qu.:  0.00   1st Qu.:  0.00  
 Mode  :character   Y:1340     Median :  0.00   Median : 25.00  
                               Mean   : 94.24   Mean   : 46.66  
                               3rd Qu.:168.00   3rd Qu.: 68.00  
                               Max.   :857.00   Max.   :547.00  
                                                                
 EnclosedPorch    ThreeSsnPorch     ScreenPorch        PoolArea      
 Min.   :  0.00   Min.   :  0.00   Min.   :  0.00   Min.   :  0.000  
 1st Qu.:  0.00   1st Qu.:  0.00   1st Qu.:  0.00   1st Qu.:  0.000  
 Median :  0.00   Median :  0.00   Median :  0.00   Median :  0.000  
 Mean   : 21.95   Mean   :  3.41   Mean   : 15.06   Mean   :  2.759  
 3rd Qu.:  0.00   3rd Qu.:  0.00   3rd Qu.:  0.00   3rd Qu.:  0.000  
 Max.   :552.00   Max.   :508.00   Max.   :480.00   Max.   :738.000  
                                                                     
    PoolQC             Fence           MiscFeature           MiscVal        
 Length:1460        Length:1460        Length:1460        Min.   :    0.00  
 Class :character   Class :character   Class :character   1st Qu.:    0.00  
 Mode  :character   Mode  :character   Mode  :character   Median :    0.00  
                                                          Mean   :   43.49  
                                                          3rd Qu.:    0.00  
                                                          Max.   :15500.00  
                                                                            
     MoSold           YrSold        SaleType    SaleCondition    SalePrice     
 Min.   : 1.000   Min.   :2006   WD     :1267   Abnorml: 101   Min.   : 34900  
 1st Qu.: 5.000   1st Qu.:2007   New    : 122   AdjLand:   4   1st Qu.:129975  
 Median : 6.000   Median :2008   COD    :  43   Alloca :  12   Median :163000  
 Mean   : 6.322   Mean   :2008   ConLD  :   9   Family :  20   Mean   :180921  
 3rd Qu.: 8.000   3rd Qu.:2009   ConLI  :   5   Normal :1198   3rd Qu.:214000  
 Max.   :12.000   Max.   :2010   ConLw  :   5   Partial: 125   Max.   :755000  
                                 (Other):   9                                  


```


#### REFERENCES:

Access to the database can be gained from the Ames site (http://www.cityofames.org/assessor/) by clicking on “property search” or by accessing the Beacon (http://beacon.schneidercorp.com/Default.aspx) website and inputting Iowa and Ames in the appropriate fields. A city map showing the location of all the neighborhoods is also available on the Ames site and can be accessed by clicking on “Maps”  and then “Residential Assessment Neighborhoods (City of Ames Only)”.


### Solution

A solution to this problem is to use one of the Regression models Linear Regression, Decision Tree Regressor, KNN Regressor or Support Vector Machine Regressor, to predict the Sale Price.


### Benchmark

Let's use the average Sale Price as our naive benchmark.


### Performance Metrics

To measure the accuracy with which Sale Price is predicted, let's use Root Mean Square Error (RMSE).
