from pytorch.review_vectorizer import ReviewVectorizer
from pytorch.vocabulary import Vocabulary

v = Vocabulary()
v.add_token("a")
v.add_token("b")

rv = ReviewVectorizer(v, v)
res = rv.vectorize("b b ")
print(res)
