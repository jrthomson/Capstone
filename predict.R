library(NLP)
library(tm)
library(stringr)
library(readr)
library(ngram)
library(RWeka)

# Read the data

twit <- readLines("./Coursera/Capstone/en_US.twitter.txt",  = FALSE, skipNul = TRUE)
blog <- readLines("./Coursera/Capstone/en_US.blogs.txt", warn = FALSE, skipNul = TRUE)
news <- readLines("./Coursera/Capstone/en_US.news.txt", warn = FALSE, skipNul = TRUE)

# Sample and cleaning the data

# Sample the data 

set.seed(98765)
sample <- c(sample(twit, length(twit) * 0.001),
            sample(blog, length(blog) * 0.001),
            sample(news, length(news) * 0.001))

# Load the data as a corpus

corpus <- Corpus(VectorSource(sample))

# Text transformation

# Replacing "/", "@" and "|" with space:

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
corpus <- tm_map(corpus, toSpace, "/")
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, "\\|")

# Cleaning the data

# Convert the text to lower case
corpus <- tm_map(corpus, content_transformer(tolower))
# Remove numbers
corpus <- tm_map(corpus, removeNumbers)
# Remove english common stopwords
corpus <- tm_map(corpus, removeWords, stopwords("english"))
# Remove punctuations
corpus <- tm_map(corpus, removePunctuation)
# Eliminate extra white spaces
corpus <- tm_map(corpus, stripWhitespace)

# Convert the corpus to a data frame

corpusdf <- data.frame(text = unlist(sapply(corpus, '[', 'content')), stringsAsFactors = F)

# Create N-Grams  
ug <- NGramTokenizer(corpusdf, Weka_control(min = 1, max = 1))
bg <- NGramTokenizer(corpusdf, Weka_control(min = 2, max = 2))
tg <- NGramTokenizer(corpusdf, Weka_control(min = 3, max = 3))
   
# Convert N-Grams to DataFrames 
ugdf <- data.frame(table(ug))
bgdf <- data.frame(table(bg))
tgdf <- data.frame(table(tg))

# Add header lables to the dataframes       
names(ugdf) <- c("Token", "Freq")
names(bgdf)  <- c("Token", "Freq")
names(tgdf) <- c("Token", "Freq")
   
# Select significant N-Grams for use in Shiny application
ugdf <- as.character(ugdf$Token[1:300000])
bgdf <- as.character(bgdf$Token[1:300000])
tgdf <- as.character(tgdf$Token[1:300000])
   
# Save N-Grams for use in Shiny application
save(ugdf, file = "./ugdf.RData")
save(bgdf, file = "./bgdf.RData")
save(tgdf, file = "./tgdf.RData")