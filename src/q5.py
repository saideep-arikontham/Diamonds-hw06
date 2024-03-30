import pandas as pd
from readit import convert_log, read_diamonds, five_fold_cross_val

df = convert_log(read_diamonds(),['carat','price'])

five_fold_cross_val(df, 'cut')
five_fold_cross_val(df, 'color')
five_fold_cross_val(df, 'clarity')
