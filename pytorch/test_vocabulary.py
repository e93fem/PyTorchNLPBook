from pytorch.vocabulary import Vocabulary

v = Vocabulary()
aid = v.add_token("a")
bid = v.add_token("b")
print(aid)
print(bid)


print(v.lookup_token("b"))
print(v.lookup_index(2))
