import cnc_input
class input_handler:
    def __init__(self, jsonfilename):
        self.target_metrices = cnc_input.main(['-i', jsonfilename])
    def A_walkonchip(self):
        allsteps = []
        for matrix in self.target_metrices:
            allsteps.extend((input_handler.sides_in_matrix(matrix[0], matrix[1], matrix[2])))
        return allsteps


    def sides_in_matrix(matrix, startpoint_x, startpoint_y):
        sides_of_matrix = ((startpoint_x, startpoint_y), (startpoint_x + len(matrix), startpoint_y) \
                               , (startpoint_x, startpoint_y + len(matrix[0]),
                                  (startpoint_x + len(matrix), startpoint_y + len(matrix[0]))))
        '''
        for i in range(1, len(matrix),6):
            for j in range(1,len(matrix[0]),6):
                walkpoints.append((startpoint_x + i, startpoint_y + j))
        return walkpoints'''
        return sides_of_matrix


    def package_points(self):
        X_all = []
        print(self.target_metrices[1])
        for matrix in self.target_metrices[0]:
            x = matrix[1]
            y = matrix[2]
            rectangle = matrix[0]
            if len(rectangle) > len(rectangle[0]):
                long_side = len(rectangle)
            else:
                long_side = len(rectangle[0])
            if (int(long_side / self.target_metrices[1]) % 2 == 0):
                corners = [[x, y], [x + len(rectangle), y],\
                           [x, y + len(rectangle[0])], [x + len(rectangle), y + len(rectangle[0])]]
            else:
                corners = [[x, y], [x + len(rectangle), y + len(rectangle[0])],\
                           [x + len(rectangle), y], [x, y + len(rectangle[0])]]
            X_all.append(corners)
        return X_all
    def point_scale(self):
        X_all = []
        for matrix in self.target_metrices[0]:
            x = matrix[1]
            y = matrix[2]
            rectangle = matrix[0]
            if len(rectangle) > len(rectangle[0]):
                long_side = len(rectangle)
            else:
                long_side = len(rectangle[0])   
            if(int(long_side / self.targetmatrices[1] % 2 == 0)):
                odd_even = 0
            else:
                odd_even = 1
            X_all.append([x,y,len(rectangle),len(rectangle[0]),odd_even])
        return X_all
    def every_point(self):
        X_all = []
        for matrix in self.target_metrices[0]:
            rectangle = matrix[0]
            x_lu = matrix[1]
            y_lu = matrix[2]
            x_ru = x_lu + len(rectangle)
            y_ru = y_lu
            x_ld = x_lu
            y_ld = y_lu + len(rectangle[0])
            x_rd = x_ru 
            y_rd = y_ld 
            
            X_all.extend([[x_lu,y_lu],[x_ru,y_ru],[x_rd,y_rd],[x_ld,y_ld]])
        return X_all
    def outcorner_getout(self,rectangle_inf):# horizontal line = row
        import torch
        feature = torch.Tensor([])
        # is odd? is row?
        for inf in rectangle_inf:
            rectangle = self.target_metrices[0][int(inf)][0]
            index = 4*int(inf)
            corner = inf - int(inf)
            if len(rectangle) > len(rectangle[0]): # is column the long side?
                long_side = len(rectangle)
                if (int(long_side / self.target_metrices[1]) % 2 == 0): # take even times to spray 
                    if corner == 0: #is a left up corner, outcorner = left down
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 1,index + 2])),0) #append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up , out = right down
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 0,index + 3])),0)
                    elif corner == 0.5: #is a right down, out = right up
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 3])),0)
                    else:               # is a left down, out = left up 
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 2])),0)
                else:    #take odd time to spray
                    if corner == 0: #is a left up corner
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 1,index + 3])),0) #append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up 
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 0,index + 2])),0)
                    elif corner == 0.5: #is a right down
                        feature = torch.cat((feature,torch.Tensor([index + 0,index + 1,index + 3])),0)
                    else:               # is a left down
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 2])),0)
                        
            else:
                long_side = len(rectangle[0]) #row = long side
                if (int(long_side / self.target_metrices[1]) % 2 == 0): # take even times to spray   
                    if corner == 0: #is a left up corner
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 2,index + 3])),0)#append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up 
                        feature = torch.cat((feature,torch.Tensor([index + 0,index + 2,index + 3])),0)
                    elif corner == 0.5: #is a right down
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 0,index + 1])),0)
                    else:               # is a left down
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 0,index + 1])),0) 
                else:
                    if corner == 0: #is a left up corner
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 1,index + 3])),0) #append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up 
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 0,index + 2])),0)
                    elif corner == 0.5: #is a right down
                        feature = torch.cat((feature,torch.Tensor([index + 0,index + 1,index + 3])),0)
                    else:               # is a left down
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 2])),0)     
        feature = torch.reshape(feature,(256,3))                
        return feature    
          
            
        
'''
test = cnc_input.main(['-i', 'right_chip.json'])
print(test[9][0])
plt.imshow(test[9][0], cmap=plt.get_cmap('gray'))
A_walkonchip(test)
print(package_points(test))
'''