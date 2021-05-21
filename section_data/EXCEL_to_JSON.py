# # Excel to JSON
# The following code converts the AISC shapes databse excel file to JSON.
# 
# The original file contains two sections per row. Manual processing must be done before the execution of this code, to create a modified file that contains only one section per row.
# 
# To decrease the lookup time for the subsequent analysis, we only extract data for specific section types.

import pandas as pd
import json

contents = pd.read_excel("aisc-shapes-database-v15.0_modified.xlsx", sheet_name="Database v15.0")


# instantiate an empty list to store the dictionaries
list_of_dicts = []
for i in range(len(contents)):  # for each row

    # turn row into a dictionary
    dct = dict(contents.loc[i])
    
    # we only want data for specific section types
    if dct["Type"] not in ["W"]:
        continue

    # filter out key-value pairs
    # where value = '-'
    # and redundant keys
    new_dct = dict()
    for (key, value) in dct.items():
        if (value != "–" and 
            key not in ["twdet/2", "bf/2tf", "h/tw"]):
            new_dct[key] = value
        # add it to the list
        list_of_dicts.append(new_dct)    

# save to a JSON file

with open("sections.json", "w") as f:
    f.write(json.dumps(list_of_dicts))

