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

print("\nFINAL RESPONSE:")
print(response)