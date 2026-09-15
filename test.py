# from tools.overview import get_overview
# import numpy as np
# import pandas as pd

# df = pd.DataFrame({
#     " A": [1,np.nan,5,4,7],
#     "B": [1,2,3,4,5],
#     "C":['A','B','C','D',np.nan]    
# }) 
# print(get_overview(df,n=4)) 

# from ollama import chat

# response = chat (

#     model = 'phi3:mini',
#     messages = [{
#         'role': 'user',
#         'content':'what is the mean of 10 and 20'
#     }
    
#     ]
    
# )

# print (response.message.content)


import pandas as pd
from agent.agent import MiniDataAnalyst


df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [10, 20, 30, 40, 50],
    "C": ["A", "B", "A", "C", "B"]
})

agent = MiniDataAnalyst(df)

response = agent.run(
    "Perform a basic analysis of this dataset."
)

print(response)