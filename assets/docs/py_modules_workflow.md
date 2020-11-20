### GUDELINES
1. please use typing and commenting (Not only Maxime likes it !!) 
1. Function with default (reccomended values) already assigned to main attributes.
1. In the end we should comply **at least** with required features and evalution critera (see Trello)
1. Feel free to provide additional details useful for the team in this file **without removing existing lines**. 

Thanks for your hard work so far !! :) 

e.g.
```Python
from typing import List
def replace_null( df: pd.DataFrame,
                   x: List[str] = ["facades_number", "a column"]) -> pd.DataFrame:
    """
    List: only columns to be considered for repalcement
    return
    """
    ...
    return df_no_null
```

### TASK LEADERS
FM
data cleaning
IN df
OUT df2
Ppt: data cleaning issues and how solved (mainly txt)

AH
Text to Number conversion & Features
IN: df2
OUT: X_train, y_train
PPT: Features engineering, Used methods description (text)

JK
model selection & apply
IN: X_train, y_train
OUT: scikit ??
(info how to extract dict of parameters)
PPT: Gradient descent, Validation curve, Learning curve

MD
Model evaluation 
IN: scikit, df, 
OUT 
PPT Root mean square error, table





