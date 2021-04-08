#Makes data samples and tests them
import latest_autoencoder as la
import random
import math
import time
import csv
random.seed(0)
print("hello")


#Data set based on 2 variables, mostly a
def makedata1(amount=100):
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

def test(data, vdata, tdata, latent_dim=2, mapping_layers=4, epochs=1):
    visible_dim = data.shape[1]
    ac = la.myAutoencoder(visible_dim=visible_dim,
                          latent_dim=latent_dim,
                          mapping_layers=mapping_layers)
    sqa = la.sequentialAutoencoder(visible_dim=visible_dim,
                                   latent_dim=latent_dim,
                                   mapping_layers=mapping_layers)

    ac.train(data, vdata, epochs=epochs)
    sqa.train(data, vdata, epochs=epochs)

    sqr = (sqa.eval(tdata).numpy())
    acr = ((ac.eval(tdata)).numpy())
    return sqr, acr

x = makedata1()
sqr, acr = test(x,x,x,epochs=10)
#sqr, acr = test(makedata1(), makedata1(), makedata1(), epochs=10)
#sqr2, acr2 = test(makedata2(), makedata2(), makedata2(), epochs=10)

print("Sequential: " + str(sqr))
print("Parallel: " + str(acr))
#print("Sequential: " + str(sqr2))
#print("Parallel: " + str(acr2))


#list the mse for each set of epochs
def mse_list(data,vdata,tdata,model=la.myAutoencoder,
             latent_dim=2, mapping_layers=4, epochs=1, count=10):
    visible_dim = data.shape[1]
    mse = []
    ac = model(visible_dim=visible_dim,
               latent_dim=latent_dim,
               mapping_layers=mapping_layers)
    for i in range(count):
        mse.append(ac.eval(tdata).numpy())
        ac.train(data=data, vdata=vdata, epochs=epochs)
    return mse



#Turn horizontal lists into average and stdeviation
def getmean(lists):
    mean = []
    stdev = []
    data = la.np.transpose(lists)
    for d in data:
        mean.append(la.np.mean(d))
        stdev.append(la.np.std(d))
    return mean, stdev


#Make a csv file with mean and stdev for both models
def makecsv(dataset, num_coders, count=2):
    data = dataset()
    vdata = dataset()
    tdata = dataset()
    ac_lists = []
    sq_lists = []
    for i in range(num_coders):
        ac_lists.append(mse_list(data,vdata,tdata,
                                 model=la.myAutoencoder,
                                 latent_dim=2, mapping_layers=4,
                                 epochs=10, count=count))
        sq_lists.append(mse_list(data,vdata,tdata,
                                 model=la.sequentialAutoencoder,
                                 latent_dim=2, mapping_layers=4,
                                 epochs=10, count=count))
    
    with open(time.strftime(str(dataset).split()[1]+'results%H%M.csv'), mode='w') as file1:
        writer1 = csv.writer(file1, delimiter=',', quotechar='"',
                             lineterminator= '\n', quoting=csv.QUOTE_MINIMAL)
           
        acm, acs = getmean(ac_lists)
        acm.insert(0,"Standard mean")
        acs.insert(0, "Standard stdev")
        sqm, sqs = getmean(sq_lists)
        sqm.insert(0, "Seq mean")
        sqs.insert(0, "Seq stdev")
        writer1.writerow(acm)
        writer1.writerow(acs)
        writer1.writerow(sqm)
        writer1.writerow(sqs)
        
#makecsv(makedata1, 10)

