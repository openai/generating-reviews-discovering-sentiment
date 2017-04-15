from encoder import Model
mdl = Model()

base = "I couldn’t figure out"
print("\'%s\'... --> (argmax sampling):" % base)

#Overriden values are slightly on the extreme on either end of 
#the sentiment's activation distribution
positive = mdl.generate_sequence(base, override={2388 : 1.0})
print("Positive sentiment (1 sentence): " + positive)
negative = mdl.generate_sequence(base, override={2388 : -1.5}, len_add = 100)
print("\nNegative sentiment (+100 chars):" + negative + '...')


n = 3
print("\n\n\'%s\'... --> (weighted samples after each word):" % base)

print("Positive sentiment (%d examples, 2 sentences each):" %n)
for i in range(n):
    positive = mdl.generate_sequence(base, override={2388 : 1.0}, len_add = '..', sampling = 1)
    print("(%d)%s" % (i, positive[1:]))
    
print("\nNegative sentiment (%d examples, 2 sentences each):" %n)
for i in range(n):
    positive = mdl.generate_sequence(base, override={2388 : -1.5}, len_add = '..', sampling = 1)
    print("(%d)%s" % (i, positive[1:]))
  

print("\n\n\'%s\'... --> (weighted samples after each character):" % base)

neutral = mdl.generate_sequence(base, len_add = '...', sampling = 2)
print("Sentiment not influenced (3 sentences):" + neutral)
neutral = mdl.generate_sequence(base, override={2388 : 0.0}, len_add = '...', sampling = 2)
print("\nSentiment fixed to 0 (3 sentences):" + neutral)
negative = mdl.generate_sequence(base, override={2388 : -1.0}, len_add = '...', sampling = 2)
print("\nSligthly negative sentiment (3 sentences):" + negative)