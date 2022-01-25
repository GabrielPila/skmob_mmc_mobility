import skmob
import seaborn as sns

print(dir(skmob))
titanic = sns.load_dataset('titanic')
print(titanic.head(20))
titanic.to_parquet('titanic.parquet')