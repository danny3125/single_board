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
    def is_odd_is_row(self,point_Lu,point_Ru,point_Rd,point_Ld):
        if (point_Lu[0] - point_Ru[0]) > (point_Ld[1] - point_Lu[1]):
            long_side = point_Lu[0] - point_Ru[0]
            if (int(long_side % self.target_metrices[1]) == 0):
                return (0,0)
            else:
                return (1,0)
            
        else:
            long_side = point_Ld[1] - point_Lu[1]
            if (int(long_side % self.target_metrices[1]) == 0):
                return (0,1)
            else:
                return (1,1)            
            
        
'''
test = cnc_input.main(['-i', 'right_chip.json'])
print(test[9][0])
plt.imshow(test[9][0], cmap=plt.get_cmap('gray'))
A_walkonchip(test)
print(package_points(test))
'''