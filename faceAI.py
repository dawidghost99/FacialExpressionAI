"""
import os
import csv
entries = os.listdir('img/')



with open('results.csv', 'w') as file:
   writer = csv.writer(file)
   for x in entries:
        
       s = ''.join([i for i in x if not i.isdigit()])
       writer.writerow([s,x])
"""

import NN

NN.NeuralNet(64,10,0)
