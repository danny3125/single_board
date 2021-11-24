import cnc_input
import json
import matplotlib.pyplot as plt


def A_walkonchip(matrices):
    allsteps = []
    for matrix in matrices:
        allsteps.extend((sides_in_matrix(matrix[0], matrix[1], matrix[2])))
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


def package_points(matrices):
    X_all = []
    for matrix in matrices:
        x = matrix[1]
        y = matrix[2]
        rectangle = matrix[0]
        if len(rectangle) > len(rectangle[0]):
            long_side = len(rectangle)
        else:
            long_side = len(rectangle[0])
        if (int(long_side / 6) % 2 == 0):
            corners = [((x, y), (x + len(rectangle), y)),
                       ((x, y + len(rectangle[0])), (x + len(rectangle), y + len(rectangle[0])))]
        else:
            corners = [(((x, y), (x + len(rectangle), y + len(rectangle[0])))),
                       ((x + len(rectangle), y), (x, y + len(rectangle[0])))]
        X_all.append(corners)
    return X_all


test = cnc_input.main(['-i', 'right_chip.json'])
print(test[9][0])
plt.imshow(test[9][0], cmap=plt.get_cmap('gray'))
A_walkonchip(test)
print(package_points(test))
