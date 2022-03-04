import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cleanweekends', '-cw', type = bool, help = "Flag to delete the datapoints of the weekends" )
parser.add_argument('--cleanoutliers','-co', type = bool, help = "Flag to delete the outlier datapoints")

args = parser.parse_args()

flag_clean_weekends = args.cleanweekends
flag_clean_outliers = args.cleanoutliers


#print(flag_clean_weekends)
#print(flag_clean_outliers)

if flag_clean_outliers or flag_clean_weekends:
    print(1)
else:
    print(2)    