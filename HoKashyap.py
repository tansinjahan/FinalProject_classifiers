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

    return a_k_1,b_k_1,Y


filepath1 = t3.currentFilePath('training_class1.csv')
trainingset_class1 = np.genfromtxt(filepath1, dtype=float, delimiter=',')
trainingset_class1 = t3.getSelectedColumns(trainingset_class1, (2, 10, 11, 14, 17, 18))

filepath2 = t3.currentFilePath('training_class2.csv')
trainingset_class2 = np.genfromtxt(filepath2, dtype=float, delimiter=',')
trainingset_class2 = t3.getSelectedColumns(trainingset_class2, (2, 10, 11, 14, 17, 18))

a,b,Y = func_call(trainingset_class1,trainingset_class2)
print("This is array of a and b and Y ", a,b,Y)
print("Shape of Y", Y.shape)
print("Shape of a", a.shape)
print("Shape of b", b.shape)

def HoKashClassifier(a,Y):
        trupositive = 0
        trunegative = 0
        classificationPoints = np.dot(a,Y)
        if classificationPoints > 0:
            trupositive = trupositive +1
        else:
            trunegative = trunegative +1

        return trupositive,trunegative

def k_fold_cross_validation(X, K):
	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation

def k_fold(class_points,k,a):
    acc = 0
    for training_x1, validation_x1 in k_fold_cross_validation(class_points, k):
        accuracy = 0
        for i in range(0, len(validation_x1)):
            temp = np.transpose(a)
            temp1= np.dot(temp,validation_x1[i])
            if temp1 > 0 and validation_x1[i][0] > 0 :
                accuracy = accuracy +1
        acc = acc + (accuracy/float(len(validation_x1)))*100
    return (acc/k)

acc = k_fold(Y,5,a)
print("This is the accuracy(5 fold) before diagonalization", acc)

def plot_HoKashyap(a,b):
    plot_x = np.array([])
    plot_y = np.array([])
    for x in range(-60,10,1):
        a_temp = a[1]
        b_temp = a[0]
        y_temp = np.multiply(a_temp,np.asarray(x))
        y = y_temp + b_temp
        plot_x = np.append(plot_x,np.asarray(x))
        plot_y = np.append(plot_y,np.asarray(y))
    #plot.generate_plot_for_X(trainingset_class1, trainingset_class2)
    return plot_x,plot_y

plot_x, plot_y = plot_HoKashyap(a,b)
plot.generate_plot_for_HoKashyap(trainingset_class1,trainingset_class2,plot_x,plot_y,'before')

filepath3 = t3.currentFilePath('diag_data_V1.csv')
dia_tr_V1 = np.genfromtxt(filepath3, dtype=float, delimiter=',')
filepath4 = t3.currentFilePath('diag_data_V2.csv')
dia_tr_V2 = np.genfromtxt(filepath4, dtype=float, delimiter=',')

a_V,b_V,Y_V = func_call(dia_tr_V1,dia_tr_V2)

acc_v = k_fold(Y_V,5,a_V)
print("This is the accuracy(5 fold) after diagonalization", acc_v)

plot_v_x, plot_v_y = plot_HoKashyap(a_V,b_V)
plot.generate_plot_for_HoKashyap(dia_tr_V1,dia_tr_V2,plot_v_x,plot_v_y,'after')

