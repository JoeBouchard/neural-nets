#Makes data samples and tests them
import latest_autoencoder as la
import random
import math

random.seed(0)
print("hello")


#Data set based on 2 variables, mostly a
def makedata1(amount = 100):
    data = []
    for i in range(amount):
        a = random.random()
        b = random.random()
        data.append([ a,
                      b,
                      a+a,
                      1/(a+1),
                      1-a
                      ])

    data = la.tf.constant(data)
    print(data.shape)
    return data

#Data set based on 2 variables, evenly distributed
def makedata2(amount=100):
    data = []
    for i in range(amount):
        a = random.random()
        b = random.random()
        data.append([   a,
                        b,
                        a+b,
                        a*b,
                        (1/(1+a)) + (1/(1+b)) 
                        ])
    data = la.tf.constant(data)
    print(data.shape)
    return data

def test(data, latent_dim=2, mapping_layers=4, epochs=1):
    visible_dim = data.shape[1]
    ac = la.myAutoencoder(visible_dim=visible_dim,
                          latent_dim=latent_dim,
                          mapping_layers=mapping_layers)
    sqa = la.sequentialAutoencoder(visible_dim=visible_dim,
                                   latent_dim=latent_dim,
                                   mapping_layers=mapping_layers)

    ac.train(data, epochs=epochs)
    sqa.train(data,epochs=epochs)

    sqr = (sqa.eval(data).numpy())
    acr = ((ac.eval(data)).numpy())
   # print("Sequential: " + str(sqr))
    #print("Parallel: " + str(acr))
    return sqr, acr

 
sqr, acr = test(makedata1(), epochs=10)
sqr2, acr2 = test(makedata2(), epochs=10)

print("Sequential: " + str(sqr))
print("Parallel: " + str(acr))
print("Sequential2: " + str(sqr2))
print("Parallel2: " + str(acr2))
