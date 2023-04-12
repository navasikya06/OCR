import h5py

filename = "checkpoints/vin.hdf5"

h5 = h5py.File(filename,'r')

list = h5.keys()

print(list)

thing1 = h5[u'model_weights']
print(thing1)

print(h5[u'optimizer_weights'])

h5.close()