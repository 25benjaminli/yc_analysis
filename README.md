## YC Analysis

Most only look at what [characteristics](https://www.kaggle.com/code/timurkhabirovich/what-yc-s-best-have-in-common) yc-funded companies [have in common](https://www.kaggle.com/code/marcelobatalhah/yc-entrepreneurs-top-companies-eda). Yet, this only gives a partial picture. What's still unknown is if there's a relationship between certain features and ability to succeed or fail *in* YC and if that can be modeled!! 

I limited the window of time to the last ten years as YC's earlier years had very sparse data with a data cutoff in fall 2024. I also added an option to narrow the analysis down to tech-related subjects specifically:
- AI
- Machine Learning
- Generative AI
- AI Assistant
- ML
etc.


Unfortunately there is a seriously limited amount of info on YC companies, especially fundraising and failure dates (unless I spend a lot of money on paid APIs). I source all the data from this [kaggle dataset](https://www.kaggle.com/datasets/sashakorovkina/ycombinator-all-funded-companies-dataset/data) with a data cutoff of 9/30/2024. I also update companies with the most recent failure data via the [yc-oss API](https://github.com/yc-oss/api?tab=readme-ov-file). The dataset only contains markers that indicate if a company is inactive, acquired, or active as of time of data retrieval. 


I have already investigated whether simple categorical variables such as tags or country can predict success but it's tricky. Due to the sheer number of tags and small dataset it's tough to make generalizations. The current goal is to structure information (e.g. tags, country, longDescription) in such a way that it can be run through a transformer (BERT) pretrained with masked language modeling. Hopefully it will capture some semantic information and a model can be trained to predict success/failure rates better than a random classifier (i.e. selects success/fail based on past proportion of success/fail). I also want to find a way to limit the influence of batch year on failure rate. 