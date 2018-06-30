
Data = [['Power Bank', 'Screen Guard' , 'Travel Charger'],
 ['Screen Guard', 'Bluetooth Headset', 'Mobile Cover'],
 ['Screen Guard','Arm Band','Mobile Cover'],
 ['Power Bank','Screen Guard','Leather Pouch'],
 ['Bluetooth Headset', 'Power Bank' , 'Mobile Cover']]


import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori

###Start code here
oht = OnehotTransactions()
oht_ary = oht.fit(Data).transform(Data)
dataFrame = pd.DataFrame(oht_ary, columns=oht.columns_)
frequent_itemsets = apriori(dataFrame, min_support=0.1, use_colnames=True)
###End code(approx 4 lines)

from mlxtend.frequent_patterns import association_rules
###Start code here
association_rule = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(association_rule)   
###End code(approx 2 lines)
''' 
                          antecedants                   consequents  \
0                          (Arm Band)                (Screen Guard)   
1        (Travel Charger, Power Bank)                (Screen Guard)   
2      (Travel Charger, Screen Guard)                  (Power Bank)   
3                    (Travel Charger)    (Power Bank, Screen Guard)   
4                 (Bluetooth Headset)                (Mobile Cover)   
5                     (Leather Pouch)                  (Power Bank)   
6   (Bluetooth Headset, Screen Guard)                (Mobile Cover)   
7            (Arm Band, Screen Guard)                (Mobile Cover)   
8            (Arm Band, Mobile Cover)                (Screen Guard)   
9                          (Arm Band)  (Mobile Cover, Screen Guard)   
10                   (Travel Charger)                  (Power Bank)   
11                   (Travel Charger)                (Screen Guard)   
12                    (Leather Pouch)                (Screen Guard)   
13                         (Arm Band)                (Mobile Cover)   
14    (Bluetooth Headset, Power Bank)                (Mobile Cover)   
15         (Power Bank, Mobile Cover)           (Bluetooth Headset)   
16        (Power Bank, Leather Pouch)                (Screen Guard)   
17      (Leather Pouch, Screen Guard)                  (Power Bank)   
18                    (Leather Pouch)    (Power Bank, Screen Guard)   

    antecedent support  consequent support  support  confidence      lift  \
0                  0.2                 0.8      0.2         1.0  1.250000   
1                  0.2                 0.8      0.2         1.0  1.250000   
2                  0.2                 0.6      0.2         1.0  1.666667   
3                  0.2                 0.4      0.2         1.0  2.500000   
4                  0.4                 0.6      0.4         1.0  1.666667   
5                  0.2                 0.6      0.2         1.0  1.666667   
6                  0.2                 0.6      0.2         1.0  1.666667   
7                  0.2                 0.6      0.2         1.0  1.666667   
8                  0.2                 0.8      0.2         1.0  1.250000   
9                  0.2                 0.4      0.2         1.0  2.500000   
10                 0.2                 0.6      0.2         1.0  1.666667   
11                 0.2                 0.8      0.2         1.0  1.250000   
12                 0.2                 0.8      0.2         1.0  1.250000   
13                 0.2                 0.6      0.2         1.0  1.666667   
14                 0.2                 0.6      0.2         1.0  1.666667   
15                 0.2                 0.4      0.2         1.0  2.500000   
16                 0.2                 0.8      0.2         1.0  1.250000   
17                 0.2                 0.6      0.2         1.0  1.666667   
18                 0.2                 0.4      0.2         1.0  2.500000   

    leverage  conviction  
0       0.04         inf  
1       0.04         inf  
2       0.08         inf  
3       0.12         inf  
4       0.16         inf  
5       0.08         inf  
6       0.08         inf  
7       0.08         inf  
8       0.04         inf  
9       0.12         inf  
10      0.08         inf  
11      0.04         inf  
12      0.04         inf  
13      0.08         inf  
14      0.08         inf  
15      0.12         inf  
16      0.04         inf  
17      0.08         inf  
18      0.12         inf  

    What is the consequent support value for Leather Pouch -> Screen Guard ?
    What is the lift value for (Arm Band, Mobile Cover)->(Screen Guard) ?
    In how many scenarios do you see 2 items (dualtons) in the antecedent set ?
    assign the above abservations to respective variable in the cell below
'''     
###Start code here
support = 0.8
lift = 1.250000
dualtons = 9
###End code(approx 3 lines)
with open("output.txt", "w") as text_file:
    text_file.write("support= %f\n" % support)
    text_file.write("lift= %f\n" % lift)
    text_file.write("dualtons= %f\n" % dualtons)

 
