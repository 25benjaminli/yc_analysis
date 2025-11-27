## YC Analysis

Most only look at what [characteristics](https://www.kaggle.com/code/timurkhabirovich/what-yc-s-best-have-in-common) yc-funded companies [have in common](https://www.kaggle.com/code/marcelobatalhah/yc-entrepreneurs-top-companies-eda). Yet, this only gives a partial picture. What's still unknown is if there's a relationship between certain features and ability to succeed or fail *within* YC and if that can be modeled!! 

I limited the window of time to the last ten years as YC's earlier years had very sparse data with a data cutoff in fall 2024. I also added an option to narrow the analysis down to tech-related subjects specifically:
- AI
- Machine Learning
- Generative AI
- AI Assistant
- ML
etc.

Unfortunately there is a seriously limited amount of info on YC companies, especially fundraising and failure dates (unless I spend a lot of money on paid APIs or somehow gain access to insider information). Therefore, I source all the data from this [kaggle dataset](https://www.kaggle.com/datasets/sashakorovkina/ycombinator-all-funded-companies-dataset/data) with a data cutoff of 9/30/2024. I also update companies with the most recent failure data via the [yc-oss API](https://github.com/yc-oss/api?tab=readme-ov-file). The dataset only contains markers that indicate if a company is inactive, acquired, or active as of time of data retrieval. 

Investigations:

1. Seeing if simple categorical variables (e.g. tags and/or country) can predict success. This is tricky due to the sheer number of tags and small dataset it's tough to make generalizations. Regardless of the model (logistic regression, gradient boosting, random forest, etc) I observe poor performance relative to a random classifier (i.e. selects success/fail based on past proportion of success/fail). 

2. Parse `longDescription` whenever available and run it through a transformer (BERT) pretrained with masked language modeling with the idea that it will capture some semantic information. Subsequently, train an algorithm to use embeddings to predict success/failure. I observe *superior performance by random forest (~0.94 f1 score) relative to the random classifier (~0.88 f1 score)*, but it's difficult to interpret what features were important for decisionmaking due to BERT's embeddings.