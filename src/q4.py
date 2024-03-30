import pandas as pd
from readit import convert_log, read_diamonds, two_feature_linear_reg

df = convert_log(read_diamonds(),['carat','price'])

two_feature_linear_reg(df, 'cut')
two_feature_linear_reg(df, 'color')
two_feature_linear_reg(df, 'clarity')
