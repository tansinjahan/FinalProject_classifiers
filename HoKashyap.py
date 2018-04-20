import numpy as np
import t3utility as t3
import matplotlib.pyplot as plt
import plot

def create_Y(class_1, class_2):
    for i in range(len(class_1)):
        np.insert(class_1,0,1,axis=1)

    for j in range(len(class_2)):
        class_2[j][:] = np.multiply([-1],class_2[j][:])
        np.insert(class_2, 0, -1, axis=1)

    Y = np.append(class_1,class_2,axis=0)
    return Y

f_L = 0.9

def find_E(Y,a,b):
    temp = np.dot(Y,a)
    temp1 = np.subtract(temp,b)
    return temp1

def update_a_b(a_old,b_old,e,f_L,Y):
    prev = np.add(e,abs(e))
    b_new = np.add(b_old,np.dot(f_L,prev))
    Y_trans = np.transpose(Y)
    Y_trans_Y = np.dot(Y_trans,Y)
    Y_trans_Y_inv = np.linalg.inv(Y_trans_Y)
    a_new1 = np.dot(Y_trans_Y_inv,Y_trans)
    a_new = np.dot(a_new1,b_new)

    return a_new,b_new

def func_call(class_1,class_2):
    Y = create_Y(class_1,class_2)
    row,col = Y.shape
    a_1 = np.ones(6)
    b_1 = np.ones(row)
    a_k_1 = a_1
    b_k_1 = b_1
    k = 1
    k_max = 200
    c = True
    while(c):
        e = find_E(Y,a_1,b_1)
        b_prev = np.sum(b_k_1)
        a_k_1, b_k_1 = update_a_b(a_k_1,b_k_1,e,f_L,Y)
        k = k+1
        e_k = np.sum(e)
        b_now = np.sum(b_k_1)
        if(e_k >= 0 or k > k_max or (b_now == b_prev)):
            c = False

    return a_k_1,b_k_1


filepath1 = t3.currentFilePath('training_class1.csv')
trainingset_class1 = np.genfromtxt(filepath1, dtype=float, delimiter=',')
trainingset_class1 = t3.getSelectedColumns(trainingset_class1, (11, 12, 13, 14, 15, 17))

filepath2 = t3.currentFilePath('training_class2.csv')
trainingset_class2 = np.genfromtxt(filepath2, dtype=float, delimiter=',')
trainingset_class2 = t3.getSelectedColumns(trainingset_class2, (11, 12, 13, 14, 15, 17))

a,b = func_call(trainingset_class1,trainingset_class2)
print("This is array of a and b ", a,b)

def plot_HoKashyap(a,b):
    plot_x = np.array([])
    plot_y = np.array([])
    for x in range(-60,10,1):
        a_temp = a[1] * a[2]
        b_temp = b[1] * b[2]
        y_temp = np.multiply(a_temp,np.asarray(x))
        y = y_temp + b_temp
        plot_x = np.append(plot_x,np.asarray(x))
        plot_y = np.append(plot_y,np.asarray(y))
    #plot.generate_plot_for_X(trainingset_class1, trainingset_class2)
    return plot_x,plot_y

plot_x, plot_y = plot_HoKashyap(a,b)
plot.generate_plot_for_HoKashyap(trainingset_class1,trainingset_class2,plot_x,plot_y)


