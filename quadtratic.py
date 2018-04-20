import numpy as np
import csv
import random
import matplotlib.pyplot as plt

# code for estimating mean and covariances using ML and Bayesian
def estimateMean_for_ML(Name_Of_class,points):

    summation = np.array([])
    a = np.sum(Name_Of_class[:,0])
    b = np.sum(Name_Of_class[:,1])
    c = np.sum(Name_Of_class[:,2])
    d = np.sum(Name_Of_class[:,3])
    e = np.sum(Name_Of_class[:,4])
    f = np.sum(Name_Of_class[:,5])
    summation = np.append(summation,a)
    summation = np.append(summation,b)
    summation = np.append(summation,c)
    summation = np.append(summation, d)
    summation = np.append(summation, e)
    summation = np.append(summation, f)
    points = float(points)
    num = np.array([points])
    mean_estimated = np.divide(summation,num)
    return mean_estimated

def estimateCovariance_forML(Name_Of_class,points,est_MLmean):
    calCov = np.zeros((6,6))
    for i in range(0,points):
        cov = Name_Of_class[i] - est_MLmean
        temp = cov
        trans_cov = temp.reshape(6,1)
        cov = cov.reshape(1,6)
        calCov = calCov + (np.dot(trans_cov,cov))
    points = float(points)
    num = np.array([points])
    cov_estimated = np.divide(calCov, num)
    return cov_estimated

def estimateMean_for_Bayesian(Name_of_class,points,actual_sigma1):
    className = Name_of_class
    sigma_nor = np.identity(6)
    mean_nor = np.zeros((6))
    for i in range(10,points+10,10):
        var1 = (1/points) * actual_sigma1 # 1/N * sigma
        var2 = var1 + sigma_nor # 1/N * sigma + sigma_nor
        inv_var2 = np.linalg.inv(var2)
        firstPart_m_n = np.dot(np.dot(var1,inv_var2),mean_nor)
        est_Mean_ML= estimateMean_for_ML(className,points)
        secondpart_m_n = np.dot(np.dot(sigma_nor,inv_var2),est_Mean_ML)
        m_n = firstPart_m_n +secondpart_m_n

    return m_n

# use sigma(covariance) of two classes from Maximum likelihood only

def beforeDiag_a_b_c(class1_mean1,class1_sigma1,class2_mean2,class2_sigma2):
    inverse_sigma1 = np.linalg.inv(class1_sigma1)
    inverse_sigma2 = np.linalg.inv(class2_sigma2)

    a = (inverse_sigma2 - inverse_sigma1)/2

    for_b1 = np.dot(np.transpose(class1_mean1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(class2_mean2),inverse_sigma2)

    b = for_b1 - for_b2

    for_c1 = np.log(1)
    for_c2 = np.log(np.linalg.det(class2_sigma2)/np.linalg.det(class1_sigma1))

    c = for_c1 +for_c2
    return a,b,c

# for finding the A,B,C domain of diagonalized points
def afterDiag_a_b_c(mean_V1,Cov_V1,mean_V2,Cov_V2):
    inverse_sigma1 = np.linalg.inv(Cov_V1)
    inverse_sigma2 = np.linalg.inv(Cov_V2)

    a = (inverse_sigma2 - inverse_sigma1)/2

    for_b1 = np.dot(np.transpose(mean_V1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(mean_V2),inverse_sigma2)

    b = for_b1 - for_b2

    for_c1 = np.log(1)
    for_c2 = np.log(np.linalg.det(Cov_V2)/np.linalg.det(Cov_V1))

    c = for_c1 +for_c2

    return a,b,c

# For question (B)
# To print in X1-X3 domain

def discriminant_function_X1X3(A,B,C):
    root1 = np.array([])
    root2 = np.array([])
    points_x1 = np.array([])

    for x1 in np.arange(-15,10,0.1):
        #for X1 - X3 domain
        p = A[2][2]
        q = ((A[0][2] * x1)+ (A[2][0] * x1) +B[2])
        r = A[0][0] * x1 *x1 + B[0] * x1 + C

        coef_array = np.array([p,q,r])
        r1, r2 = np.roots(coef_array)

        root1 = np.append(root1,r1)
        root2 = np.append(root2,r2)
        points_x1 = np.append(points_x1,[x1])

    return root1,root2,points_x1

def discriminant_function_X1X2(A,B,C):
    root1 = np.array([])
    root2 = np.array([])
    points_x1 = np.array([])

    for x1 in np.arange(-15, 20, 0.1):
        # for X1 - X2 domain
        m = A[1][1]
        n = ((A[0][1] * x1) + (A[1][0] * x1) + B[1])
        o = A[0][0] * x1 * x1 + B[0] * x1 + C

        coef_array = np.array([m, n, o])
        r1, r2 = np.roots(coef_array)

        root1 = np.append(root1, r1)
        root2 = np.append(root2, r2)
        points_x1 = np.append(points_x1, [x1])

    return root1, root2, points_x1

# parzen window calculate for a single feature
def parzen_window_calculate(Name_of_class, sigma):
    className = Name_of_class
    sample_points_first = className[:, 2]
    sample_points_first = np.sort(sample_points_first)

    l1 = sample_points_first.size

    f_x1_first = np.array([])

    random_X1 = np.random.rand(500,6)

    random_X1_first = random_X1[:, 2]
    random_X1_first = np.sort(random_X1_first)

    max_1 = np.max(random_X1_first)
    # print("random points and sample points:",random_X1_first,sample_points_first)
    mean_parzen = np.array([])
    covariance_parzen = np.array([])
    ex_mean = 0
    ex_cov = 0
    j = 0;
    for i in random_X1_first:
        temp = 0
        for x in sample_points_first:
            temp1 = 1.0 / (np.sqrt(2 * 3.1416) * sigma)
            temp2 = np.power((i - x), 2)
            temp3 = 2 * np.power(sigma, 2)
            temp4 = np.exp(-1.0 * (temp2 / temp3))
            temp5 = temp4 * temp1
            temp = temp5 + temp
        temp = temp / l1

        if (i != max_1):
            del_x = np.subtract(random_X1_first[j + 1], random_X1_first[j])
            ex_mean = ex_mean + del_x * temp * i
            # print("This is expected mean", ex_mean)
            temp_cov = np.power((i - ex_mean), 2)
            ex_cov = ex_cov + temp_cov * temp * del_x
            # print("This is expected covariance", ex_cov)
            j = j + 1
        f_x1_first = np.append(f_x1_first, temp)

    mean_parzen = np.append(mean_parzen, ex_mean)
    covariance_parzen = np.append(covariance_parzen, ex_cov)

    return f_x1_first, random_X1_first, mean_parzen, covariance_parzen

# Five fold cross validation

def k_fold_cross_validation(X, K):
	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation

true_positive_X1 = 0
true_negative_X1= 0
true_positive_V1 = 0
true_negative_V1= 0

true_positive_X2 = 0
true_negative_X2= 0
true_positive_V2 = 0
true_negative_V2= 0

def tenfold_X1(class_points,A,B,C):
    for training_x1, validation_x1 in k_fold_cross_validation(class_points,5):
        for i in range(0,len(validation_x1)):
            global true_positive_X1
            global true_negative_X1
            value = ((np.dot(np.dot(validation_x1[i],A), np.transpose(validation_x1[i]))) + (np.dot(B, np.transpose(validation_x1[i]))) + C)
            if value > 0:
                true_positive_X1 = true_positive_X1 + 1
            else:
                true_negative_X1 = true_negative_X1 + 1

    return true_positive_X1, true_negative_X1

def tenfold_X2(class_points,A,B,C):
    for training_x2, validation_x2 in k_fold_cross_validation(class_points,5):
        for i in range(0,len(validation_x2)):
            global true_positive_X2
            global true_negative_X2
            value = ((np.dot(np.dot(validation_x2[i], A), np.transpose(validation_x2[i]))) + (np.dot(B, np.transpose(validation_x2[i]))) + C)
            if value < 0:
                true_positive_X2 = true_positive_X2 + 1
            else:
                true_negative_X2 = true_negative_X2 + 1

    return true_positive_X2, true_negative_X2


def tenfold_V1(class_points,A,B,C):
    for training_v1, validation_v1 in k_fold_cross_validation(class_points,5):
        for i in range(0,len(validation_v1)):
            global true_positive_V1
            global true_negative_V1
            value = ((np.dot(np.dot(validation_v1[i],A), np.transpose(validation_v1[i]))) + (np.dot(B, np.transpose(validation_v1[i]))) + C)
            if value > 0:
                true_positive_V1 = true_positive_V1 + 1
            else:
                true_negative_V1 = true_negative_V1 + 1

    return true_positive_V1, true_negative_V1

def tenfold_V2(class_points,A,B,C):
    for training_v2, validation_v2 in k_fold_cross_validation(class_points,5):
        for i in range(0,len(validation_v2)):
            global true_positive_V2
            global true_negative_V2
            value = ((np.dot(np.dot(validation_v2[i], A), np.transpose(validation_v2[i]))) + (np.dot(B, np.transpose(validation_v2[i]))) + C)
            if value < 0:
                true_positive_V2 = true_positive_V2 + 1
            else:
                true_negative_V2 = true_negative_V2 + 1

    return true_positive_V2, true_negative_V2


def accuracy(TP1,TN1,TP2,TN2):
    a = (TP1+TP2)
    b = (TP1+TN1+TP2+TN2) * 1.0
    c = float (a/b)
    c = c * 100
    return c
