import random
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Rahul', 'Abhi', 'Jon', 'Ram', 'Jay'],
    'Subject': [random.choice(['Math', 'Science', 'English']) for _ in range(10)],
    'Score': [random.randint(50, 100) for _ in range(10)],
    'Grade': [None for _ in range(10)]
}

df = pd.DataFrame(data)

# 1st part
def get_grade(score):
     return {
          'grade': 'A' if score >= 90 else
                   'B' if score >= 80 else
                   'C' if score >= 70 else
                   'D' if score >= 60 else
                   'E'
     }['grade']

df['Grade'] = df['Score'].apply(get_grade)

# 2nd part
print(df.sort_values(['Score'], ascending=False))

# 3rd part
sub_avg = df.groupby('Subject', as_index=False)['Score'].mean().round(2)

print(sub_avg)

# 4th part
def pandas_filter_pass(df):
     ndf = df.loc[(df['Grade'] == 'A') | (df['Grade'] == 'B')]
     ndf = ndf.reset_index(drop = True)

     return ndf