# Sentiment-and-Review-Analysis
The first script is for sentiment analysis on Google. The second script is for fake review detection on reviews. Both use transformer libraries on Hugging Face

For the Sentiment.py tool, 

      from transformers import pipeline
      import numpy as np
      import pandas as pd
      import seaborn as sns
      import matplotlib.pyplot as plt
      from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
      from sklearn.model_selection import train_test_split


      # create pipeline for sentiment analysis
      classification = pipeline('sentiment-analysis')
      type(classification)

      # read data into DataFrame
      df = pd.read_csv("google.csv")
      # set the columns
      df.columns = ['Links', 'Name', 'Time', 'Review', 'Owner Response']
      # drop null values
      df = df.dropna()
      # print first 5 rows
      df.head()

      results = df['Review'].apply(classification)

      df['Sentiment'] = [result[0]['label'] for result in results]
      df['Score'] = [result[0]['score'] for result in results]

      df.to_csv('google_sentiment.csv', columns=['Name', 'Time', 'Review', 'Sentiment', 'Owner Response'], index=False)

      # filter DataFrame by negative sentiment and save as new csv file
      df_negative = df.query('Sentiment == "NEGATIVE"')
      df_negative.to_csv('google_negative.csv', columns=['Name', 'Time', 'Review', 'Sentiment', 'Owner Response'], index=False)

      print ("done")



The first line imports the pipeline function from the transformers library, which is a high-level API for using pre-trained models for natural language processing tasks

The next four lines import some other libraries that are used for numerical computation, data analysis, data visualization, and machine learning

The next line creates a pipeline for sentiment analysis using the default pre-trained model, which is distilbert-base-uncased-finetuned-sst-2-english. This model is a distilled version of BERT, a state-of-the-art model for natural language understanding, and it is fine-tuned on a dataset of movie reviews

The next line prints the type of the pipeline object, which is transformers.pipelines.TextClassificationPipeline.

The next line reads a csv file named google.csv into a DataFrame object using the pandas library. A DataFrame is a two-dimensional data structure that can store data in rows and columns.

The next line sets the column names of the DataFrame to Links, Name, Time, Review, and Owner Response.

The next line drops any rows that have missing values in any of the columns using the dropna() method.

The next line prints the first five rows of the DataFrame using the head() method.

The next line applies the sentiment analysis pipeline to each review in the DataFrame using the apply() method. This returns a list of dictionaries, where each dictionary contains the label and the score of the sentiment for each review.

The next two lines create two new columns in the DataFrame, Sentiment and Score, and assign them the values from the results list using list comprehension.

The next line saves the DataFrame with the new columns as a new csv file named google_sentiment.csv using the to_csv() method. The columns to be saved are specified as a list, and the index is set to False to avoid saving the row numbers.

The next line filters the DataFrame by selecting only the rows that have a negative sentiment using the query() method. This returns a new DataFrame with only the negative reviews.
The next line saves the new DataFrame as another csv file named google_negative.csv using the same method as before.

The last line prints "done" to indicate that the script has finished running.



For the FakeReview.py tool, 

      # Import the required libraries
      import pandas as pd
      from torch.utils.data import Dataset
      from transformers import pipeline

      # Create pipeline for fake review detection
      fake_review_detection = pipeline('text-classification', model='astrosbd/fake-reviews-distilbert')

      # Read the csv file into a DataFrame
      df = pd.read_csv('google.csv')

      # Drop any rows that have missing values
      df = df.dropna()

      # Shuffle the DataFrame for randomness
      df = df.sample(frac=1, random_state=42)

      # Initialize empty lists to store results
      scores = []
      labels = []

      # Loop over the rows of the DataFrame
      for index, row in df.iterrows():
          # Get the review from the current row
          review = row['Review']
          # Get the model's output for the current review
          result = fake_review_detection(review)
          # Extract the score and label from the output
          score = result[0]['score']
          label = result[0]['label']
          # Append the score and label to the respective lists
          scores.append(score)
          labels.append(label)

      # Add the scores and labels as new columns to the DataFrame
      df['Score'] = scores
      df['Label'] = labels

      # Save DataFrame with new columns as a new csv file
      df.to_csv('google_new.csv', columns=['Review', 'Score', 'Label'], index=False)

      print ("done")



The first three lines import the required libraries for the script. pandas is a library for data analysis and manipulation. torch.utils.data is a module for creating and handling data sets. transformers is a library for using pre-trained models for natural language processing tasks

The next line creates a pipeline for fake review detection using the text-classification task and the astrosbd/fake-reviews-distilbert model. This model is a fine-tuned version of DistilBERT, a distilled version of BERT, a state-of-the-art model for natural language understanding. The model is trained on a dataset of Amazon reviews and can classify them as fake or real

The next line reads a csv file named google.csv into a DataFrame object using the pandas library. A DataFrame is a two-dimensional data structure that can store data in rows and columns.

The next line drops any rows that have missing values in any of the columns using the dropna() method.

The next line shuffles the DataFrame for randomness using the sample() method. The frac argument specifies the fraction of rows to return, which is 1 in this case, meaning all rows. The random_state argument sets the seed for the random number generator, which is 42 in this case, meaning the same shuffle order will be used every time the script is run.

The next two lines initialize empty lists to store the scores and labels of the fake review detection results.

The next block of code loops over the rows of the DataFrame using the iterrows() method, which returns an iterator of index and row pairs. For each row, the following steps are performed:

   The review from the current row is extracted and assigned to the review variable.

   The fake review detection pipeline is applied to the review and the output is assigned to the result variable. The output is a list of dictionaries, where each dictionary contains the label and the score    of the fake review detection for the review.
   
   The score and label from the output are extracted and assigned to the score and label variables, respectively.
   
   The score and label are appended to the respective lists using the append() method.
   
The next two lines create two new columns in the DataFrame, Score and Label, and assign them the values from the lists using list assignment.

The next line saves the DataFrame with the new columns as a new csv file named google_new.csv using the to_csv() method. The columns to be saved are specified as a list, and the index is set to False to avoid saving the row numbers.

The last line prints "done" to indicate that the script has finished running.
