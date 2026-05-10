import numpy as np
class Mnistreader:

    DTYPE_LOOKUP = { #attributo di classe, tabella di lookup per assegnare i tipi corretti
        0x08: np.uint8,
        0x09: np.int8,
        0x0B: np.int16,
        0x0C: np.int32,
        0x0D: np.float32,
        0x0E: np.float64,
        }

    def __init__(self, path=None):      #costruttore
        self.path = path                #attributi di instanza
        self.magic_number = None
        self.n_dims = None
        self.data_type = None
        self.data = None
        self.load(path)




    def load(self, path):
        with open(path, 'rb') as f:     #equal to: open a stream of data called f inside a try-catch and close the stream automatically afterwards
            b = f.read()                #read the bytes #TODO fix for big files, split in chunks, use progress bar or sth

            #PADDING CHECK
            self.paddingCheck(b)

            #MAGIC NUMBER EXTRACTION
            self.magic_number = b[2]    #save the magic number

            #DATA TYPE LOOKUP
            self.data_type = self.DTYPE_LOOKUP.get(b[2]) #TODO aggiungi check se non trova il magicnumber corretto

            #SIZE EXTRACTION
            self.n_dims = b[3]
            #dimensions are in 4 bytes integers: we need to group them.
            dims=[]
            for i in range(4,4*self.n_dims+1, 4): #read groups of 4 bytes 
                dims.append(int.from_bytes(b[i:i+4],byteorder="big"))

            #DATA ALLOCATION AND READING #TODO chunks!

            a0 = 4 + self.n_dims * 4        #a0 = actual data address 0 (first address of actual data) #TODO could split HEADER and DATA directly.

            self.data = np.frombuffer(b[a0:], self.data_type).reshape(dims) #was #self.data = np.empty(dims, dtype=self.data_type)

            #DATA RETURN
            return self.data
        


    def paddingCheck(self, b):
        if not (b[0] == 0 and b[1] == 0):
            raise ValueError("Invalid Magic number padding. the first two bytes of the file should both be b'\00. are you sure the file is UBYTE?")
        

#TODO function that, given an index, returns the whole image.
#TODO a sister function that, given an index, returns the associated label.

'''
    STRUCTURE:

   magic number
   size in dimension 1
   size in dimension 2
   size in dimension 3
   ....
   size in dimension N
   data

   MAGIC NUMBER: 4 bytes long (32 bit integer)
   00 00 AA BB
   where AA codes the data type:

   0x08: unsigned byte
   0x09: signed byte
   0x0B: short (2 bytes)
   0x0C: int (4 bytes)
   0x0D: float (4 bytes)
   0x0E: double (8 bytes)

   and BB encodes the number of dimensions of the data (1 - array, 2 - matrix etc)  
'''





'''
DEMO
'''
d_path = 'DATA/train-images.idx3-ubyte'
l_path = 'DATA/train-labels.idx1-ubyte'

d = Mnistreader(d_path)
l = Mnistreader(l_path)

for i in range(28):
    for j in range(28):
        if (d.data[0][i][j] == 0):
            print('. ', end=" ")
        else:
            print('# ' , end=" ")
        #print(m.data[0][i][j], end="")
    print('\n')
print(l.data[0])



import matplotlib.pyplot as plt


fig = plt.figure(figsize=(15,15))
columns = 10
rows = 10

ax = []

for i in range(columns*rows):
    img = d.data[i]
    label = l.data[i]
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title('Label: '+str(label))
    plt.imshow(img, cmap='binary')

plt.show()
