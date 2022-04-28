# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:32:54 2019

@author: JiangZhao
"""

#from Tkinter import *
from tkinter import Tk, Frame, SUNKEN, Scrollbar, HORIZONTAL, E, W, N, S, Canvas, BOTH, ALL
# from tkFileDialog import askopenfilename
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk
#from pandas import DataFrame
import numpy as np
import h5py
from keras.models import load_model
from keras.engine.topology import Layer
import keras.backend as K

class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'channels_first':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'channels_last':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'channels_first':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'channels_last':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs


model = load_model('F:/rice_heading/抽随比例识别/densnet121nobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})
# model = load_model('densnet121nobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})
model.summary()

lrs = []#LeafRollingScore
patch_count = 0


if __name__ == "__main__":
    root = Tk()
    root.title("Picture")

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    
    
    imgFile0 = Image.open(File)
    scale = 6
    imgData = np.array(imgFile0)
    imgFile = imgFile0.resize([int(imgFile0.size[0]/scale), int(imgFile0.size[1]/scale)], resample = 1)
    
    
    img = ImageTk.PhotoImage(imgFile)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))


    #function to be called when mouse is clicked
    def func(event):
        #outputting x and y coords to console
        row = event.y
        col = event.x
        # _data: positions for data, not scaled
        row_data = scale * row
        col_data = scale * col
#        print(row_data,col_data)
        
        #get patch
        
        #patchSize = 256
        patchSize = 256
        
        
        r_data = int(patchSize/2)
        x1_data = row_data - r_data
        y1_data = col_data - r_data
        
        
        #patch = imgData[x1_data:x1_data+256, y1_data:y1_data+256, :]
        patch = imgData[x1_data:x1_data+224, y1_data:y1_data+224, :]
        #X = np.zeros([1,256,256,3])
        X = np.zeros([1,224,224,3])
        
        
        X[0] = patch
        y_temp = model.predict(X).squeeze()
        
        global lrs
        global patch_count
        lrs.append(y_temp)
        patch_count += 1
        print('panicle ratio ' + str(patch_count) + ': ' + str(y_temp))
        # visulization
        # bestColor = "#%12x%12x%12x" % (int(255-np.median(patch[:,:,0])), int(255-np.median(patch[:,:,1])), int(255-np.median(patch[:,:,2])))
        bestColor = '#FFFF00'
        r = r_data/scale
        canvas.create_rectangle(col-r, row-r, col+r, row+r, outline='#FF0000')
#        print(patch.shape)
        canvas.create_text(col-r-5, row-r-5,fill=bestColor,font="Times 10 bold",text=str(patch_count))
        canvas.create_text(col, row,fill=bestColor,font="Times 10 bold",text=str(np.round(y_temp,2)))
        
#        lrs_df = DataFrame({'Id': range(1, patch_count+1), 'Leaf-rolling score': lrs})
#        lrs_df.to_excel('./LeafRollingScore.xlsx', index=False)
        # np.savetxt('./Panicle ratio.csv', lrs, delimiter='\n')
        np.savetxt('F:/rice_heading/抽随比例识别/Panicle ratio.csv', lrs, delimiter='\n')

    #mouseclick event
    canvas.bind("<Button 1>",func)

    root.mainloop()
