# Opinion mining

## Data analysis

The data is a collection of customer reviews, extraced from Amazon and manually labelled with product features, sentiment polarity and sentiment strength. Each file contains reviews for one specific product or domain. 

Symbols used in the annotated reviews (from Customer_review_data/Readme.txt): 

  [t]: the title of the review: Each [t] tag starts a review. 
       We did not use the title information in our papers.
  xxxx[+|-n]: xxxx is a product feature. 
      [+n]: Positive opinion, n is the opinion strength: 3 strongest, 
            and 1 weakest. Note that the strength is quite subjective. 
            You may want ignore it, but only considering + and -
      [-n]: Negative opinion
  ##  : start of each sentence. Each line is a sentence. 
  [u] : feature not appeared in the sentence.
  [p] : feature not appeared in the sentence. Pronoun resolution is needed.
  [s] : suggestion or recommendation.
  [cc]: comparison with a competing product from a different brand.
  [cs]: comparison with a competing product from the same brand.
  
  ## Data preprocessing
  
  The opinion mining will be for practical reasons run per product. The reviews of a product (or domain) are first parsed and the annotations removed and store separately. Individual sentences are retained. The titles are not used. For parsing of the corpora the NLTK review reader is used [2]
  
  
  ## Product feature extraction
  
  - focusing on the explicitely mentioned features since the implicite ones are more rare.[1]
  
  
  [1] https://www.cs.uic.edu/~liub/publications/aaai04-featureExtract.pdf "Liu, Hu, Mining Opinion Features in Customer Reviews"
  [2] https://www.nltk.org/howto/corpus.html "NLTK product reviews corpus"