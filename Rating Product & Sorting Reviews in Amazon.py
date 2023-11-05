
###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the correct calculation of the points given to products after sales.
# # The solution to this problem means providing more customer satisfaction for the e-commerce site, making the product
# stand out for sellers, and a smooth shopping experience for buyers.
# Another problem is the correct ordering of the comments given to the products. Highlighting misleading comments will
# directly affect the sales of the product, causing both financial and customer loss. By solving these 2 basic problems,
# e-commerce sites and sellers will increase their sales, while customers will complete their purchasing journey without
# any problems.

###################################################

# This dataset, which contains Amazon product data, includes product categories and various metadata.
# The product with the most comments in the electronics category has user ratings and comments.
#
# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: Username
# helpful: Helpful review rating: How many good comments are there for each comment?
# reviewText: Review
# overall: Product rating
# summary: Evaluation summary
# unixReviewTime: Review time
# reviewTime: Review time Raw
# day_diff: Number of days since evaluation
# helpful_yes: Number of times the review was found helpful
# total_vote: Number of votes given to the review = total_vote is the total number of up-downs given to a comment.
# up means helpful.


###################################################
# Calculate the Average Rating Based on Current Comments and Compare it with the Existing Average Rating.
###################################################

# In the shared data set, users rated a product and made comments.
# Our aim in this task is to evaluate the given points by weighting them according to date.
# It is necessary to compare the initial average score with the date-weighted score to be obtained.

###################################################
###################################################
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import warnings
warnings.filterwarnings("ignore")
df_ = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/ödevler/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df = df_.copy()

df.head()
###################################################
# Calculate the Weighted Score Average by Date
###################################################

# Calculate the average score of the product.

df['overall'].mean()




#Calculate the weighted score average by date. Time-Based Weighted Average

# I checked if the #day_diff variable is the same thing.
df['reviewTime'].max()   #'2014-12-07'
df["reviewTime"] = pd.to_datetime(df["reviewTime"], dayfirst=True)
current_date = pd.to_datetime("2014-12-07")
df["day_fark"] = (current_date - df['reviewTime']).dt.days

#----------------------------
df.head()
df.info()
df["day_diff"].value_counts()
df["day_diff"].mean()
df["day_diff"].describe()

x = df["day_diff"].quantile(0.25) #281
y = df["day_diff"].quantile(0.50) #431
z = df["day_diff"].quantile(0.75) #601

df.loc[df["day_diff"] <= 281, "overall"].mean() * 30 / 100 + \
df.loc[(df["day_diff"] > 281) & (df["day_diff"] <= 431), "overall"].mean() * 25 / 100 + \
df.loc[(df["day_diff"] > 431) & (df["day_diff"] <= 601), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > 601), "overall"].mean() * 21 / 100



# 4.598685075638924 Time-Based Weighted Average

###The rating has decreased over time.

#df.loc[df["day_diff"] <= 281, "overall"].mean()    4.6957928802588995
#df.loc[(df["day_diff"] > 281) & (df["day_diff"] <= 431), "overall"].mean()     4.636140637775961
#df.loc[(df["day_diff"] > 431) & (df["day_diff"] <= 601), "overall"].mean()      4.571661237785016
#df.loc[(df["day_diff"] > 601), "overall"].mean()                            4.4462540716612375



###################################################
# Identify 20 Reviews to Display on the Product Detail Page for the Product. #sorting review
###################################################


###################################################
# Create the variable helpful_no
###################################################

# Note:
# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated from existing variables.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()
df.info()
df["helpful_no"].value_counts()


###################################################
# Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and Add them to the Data
###################################################
up = df["helpful_yes"]
down = df["helpful_no"]

# score_pos_neg_diff
def score_up_down_diff(up, down):
    return up - down


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

# score_average_rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is considered as the WLB score.
    - The score to be calculated is used for product ranking.
    - Note:
    If the scores are between 1-5, 1-3 is marked as negative and 4-5 is marked as positive and can be adapted to Bernoulli.
    This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()


##################################################
# Identify 20 Interpretations and Interpret the Results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df[["reviewerName","reviewText","score_average_rating","wilson_lower_bound"]].sort_values("wilson_lower_bound", ascending=False).head(20)


# most liked comment Hyoun Kim "Faluzure" [[ UPDATE - 6/19/2014 ]]So my lovely wife boug...
