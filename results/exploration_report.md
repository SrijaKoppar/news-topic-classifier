# Exploration report

## 1. Overview and schema

- Rows: 25000
- Columns: ['id', 'headline', 'short_description', 'authors', 'link', 'date', 'category', 'split_a', 'split_b']
- Unexpected extra columns: none
- Labels outside the allowed 9: none
- Allowed labels that never appear: none

Values in `split_a` (including missing):
```
split_a
train    19702
test      5298
```

Values in `split_b` (including missing):
```
split_b
train    19992
test      5008
```

## 2. Missing data

```
                   missing_or_blank    pct
column                                    
id                                0   0.00
headline                          0   0.00
short_description              2319   9.28
authors                        4351  17.40
link                              0   0.00
date                              0   0.00
category                          0   0.00
split_a                           0   0.00
split_b                           0   0.00
```

Rows with BOTH headline and description empty: 0

## 3. Duplicates

- Extra rows sharing an `id`: 0
- Extra rows identical in every column except `id`: 0
- Extra rows with identical model text (case-insensitive): 12
- Distinct texts that appear with more than one label: 1

`split_a`: 3 of 5298 test rows have text that also appears in this split's train rows; 0 of those carry a label never seen for that text in train.

`split_b`: 0 of 5008 test rows have text that also appears in this split's train rows; 0 of those carry a label never seen for that text in train.

## 4. Class balance

```
                   count    pct
category                       
POLITICS            5610  22.44
WORLD               1565   6.26
BUSINESS & MONEY    1345   5.38
TECH & SCIENCE       741   2.96
ENTERTAINMENT       4164  16.66
SPORTS               842   3.37
HEALTH & WELLNESS   4239  16.96
LIFESTYLE           5999  24.00
EDUCATION            495   1.98
```

Largest / smallest class ratio: 12.1

Class balance by `split_a`:
```
split_a            test  train  test_%  train_%
category                                       
BUSINESS & MONEY    284   1061    5.36     5.39
EDUCATION           105    390    1.98     1.98
ENTERTAINMENT       884   3280   16.69    16.65
HEALTH & WELLNESS   908   3331   17.14    16.91
LIFESTYLE          1258   4741   23.74    24.06
POLITICS           1190   4420   22.46    22.43
SPORTS              180    662    3.40     3.36
TECH & SCIENCE      154    587    2.91     2.98
WORLD               335   1230    6.32     6.24
```

Class balance by `split_b`:
```
split_b            test  train  test_%  train_%
category                                       
BUSINESS & MONEY     80   1265    1.60     6.33
EDUCATION           100    395    2.00     1.98
ENTERTAINMENT      1232   2932   24.60    14.67
HEALTH & WELLNESS   198   4041    3.95    20.21
LIFESTYLE           286   5713    5.71    28.58
POLITICS           2282   3328   45.57    16.65
SPORTS              138    704    2.76     3.52
TECH & SCIENCE       62    679    1.24     3.40
WORLD               630    935   12.58     4.68
```

## 5. Train/test splits

`split_a` counts:
```
split_a
train    19702
test      5298
```
`split_b` counts:
```
split_b
train    19992
test      5008
```

Cross-tab of split_a (rows) vs split_b (columns):
```
split_b  test  train
split_a             
test     1079   4219
train    3929  15773
```

## 6. Text lengths

```
                           count   mean   std  min    1%    5%    50%    95%    99%     max
headline chars           25000.0   58.7  18.8  3.0  16.0  26.0   60.0   88.0  101.0   150.0
headline words           25000.0    9.6   3.1  1.0   3.0   4.0   10.0   15.0   17.0    28.0
short_description chars  25000.0  113.0  82.3  0.0   0.0   0.0  120.0  264.0  320.0  1361.0
short_description words  25000.0   19.3  14.2  0.0   0.0   0.0   19.0   46.0   56.0   241.0
combined chars           25000.0  172.5  80.2  3.0  35.0  56.0  170.0  313.0  377.0  1376.0
combined words           25000.0   28.9  13.9  1.0   6.0   9.0   28.0   54.0   65.0   243.0
```

## 7. Dates

- Missing dates: 0
- Present but not YYYY-MM-DD: 0
- Earliest: 2012-01-28 00:00:00  Latest: 2022-09-20 00:00:00

Rows per year:
```
date
2012    4021
2013    4263
2014    3711
2015    3841
2016    4156
2017    2537
2018     964
2019     363
2020     417
2021     429
2022     298
```

Train/test date range for `split_a`:
```
               min        max
split_a                      
test    2012-01-28 2022-09-19
train   2012-01-28 2022-09-20
```

Rows per year by `split_a`:
```
split_a  test  train
date                
2012      875   3146
2013      902   3361
2014      790   2921
2015      786   3055
2016      866   3290
2017      527   2010
2018      208    756
2019       76    287
2020      108    309
2021      101    328
2022       59    239
```

Train/test date range for `split_b`:
```
               min        max
split_b                      
test    2017-01-01 2022-09-20
train   2012-01-28 2016-12-31
```

Rows per year by `split_b`:
```
split_b  test  train
date                
2012        0   4021
2013        0   4263
2014        0   3711
2015        0   3841
2016        0   4156
2017     2537      0
2018      964      0
2019      363      0
2020      417      0
2021      429      0
2022      298      0
```

Class share (%) per year (rows = labels, columns = years):
```
date               2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022
category                                                                           
BUSINESS & MONEY    9.5   3.8   8.0   6.4   4.3   1.9   1.1   1.9   1.7   0.9   1.0
EDUCATION           0.0   0.0   4.3   4.8   1.2   2.2   3.5   0.6   0.2   0.9   1.0
ENTERTAINMENT       4.5   6.3  14.8  26.8  21.8  21.2  32.4  26.7  25.9  24.0  25.2
HEALTH & WELLNESS  31.5  34.2  20.1   9.1   5.2   6.3   0.1   1.4   4.3   1.4   3.0
LIFESTYLE          50.0  51.7  24.5   8.5   6.4   6.6   1.2   6.1  11.3   5.4   4.7
POLITICS            0.0   0.0  15.0  26.5  42.2  47.6  42.3  34.7  48.2  52.0  39.3
SPORTS              1.2   1.3   4.5   5.9   5.0   1.7   4.6   2.8   1.9   3.5   6.0
TECH & SCIENCE      3.3   2.7   2.8   4.3   4.0   1.2   2.0   1.4   0.2   0.7   1.0
WORLD               0.0   0.0   6.1   7.6  10.0  11.4  12.8  24.5   6.2  11.2  18.8
```

## 8. Anything unusual

```
combined text under 5 words            97
short_description empty              2319
headline identical to description      16
text contains a URL                    28
text contains HTML tag/entity           0
headline ALL CAPS (10+ chars)           7
```

The 10 shortest texts (for you to eyeball):
```
                  headline short_description           category
57                   Once.                        ENTERTAINMENT
1283           Into-Me-See                    HEALTH & WELLNESS
2337                  Test                            LIFESTYLE
4649              Attendez                            LIFESTYLE
9830            Trainwreck                        ENTERTAINMENT
12416             #Anxiety                    HEALTH & WELLNESS
14511               Poetry                            EDUCATION
17489                  CUT                               SPORTS
48             Lucky Penny                        ENTERTAINMENT
4926   Chelsea's Expecting                             POLITICS
```
