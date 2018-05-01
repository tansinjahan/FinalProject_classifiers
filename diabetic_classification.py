import numpy as np
import t3utility as t3
import random_vector_generation as gn
import plot
import kNN as knn
import quadtratic as quad
#import HoKashyap as hp
import FisherDiscriminant as fd

# class 1 and class 2 points generation from random_vector_generation.py

trainingset_class1 = gn.trainingset_class1
trainingset_class2 = gn.trainingset_class2

class1_points = gn.class1_points
class1_col = gn.class1_col
class2_points = gn.class2_points
class2_col = gn.class2_col

# whole dataset with no filtering
filepath3 = t3.currentFilePath('data.csv')
trainingset_class3 = np.genfromtxt(filepath3, dtype=float, delimiter=',')
trainingset_class3 = t3.getSelectedColumns(trainingset_class3, (4, 8, 11, 14, 17, 18,20))
np.savetxt("data_mod.csv",trainingset_class3,delimiter=",")

# plotting two classes in 2-dimension
plot.generate_plot_for_X(trainingset_class1,trainingset_class2)

#--------Quadratic classifier--------------#
# estimated mean for ML
est_MLmean_class1 = gn.est_MLmean_class1
est_MLmean_class2 = gn.est_MLmean_class2
print("This is estimated(ML) Mean for class1 and class2:", est_MLmean_class1,est_MLmean_class2)

# estimated covariance for ML
est_MLcovariance_class1 = gn.est_MLcovariance_class1
est_MLcovariance_class2 = gn.est_MLcovariance_class2
print("This is estimated(ML) covariances for class1 and class2:", est_MLcovariance_class1,est_MLcovariance_class2)

# estimated mean for Bayesian
est_BLmean_class1= quad.estimateMean_for_Bayesian(trainingset_class1,class1_points,est_MLcovariance_class1)
est_BLmean_class2= quad.estimateMean_for_Bayesian(trainingset_class2,class2_points,est_MLcovariance_class2)
print ("this is estimated(BL) mean for class1 and class2",est_BLmean_class1, est_BLmean_class2)

# find quadtratic classifiers domain(A,B,C) before diagonalization
ML_A,ML_B,ML_C = quad.beforeDiag_a_b_c(est_MLmean_class1,est_MLcovariance_class1,est_MLmean_class2,est_MLcovariance_class2)


#plotting discriminant function on X1-X2 and X1-X3 domain
Root1,Root2,Points_X1_X3 = quad.discriminant_function_X1X3(ML_A,ML_B,ML_C)
Root3,Root4,Points_X1_X2 = quad.discriminant_function_X1X2(ML_A,ML_B,ML_C)

plot.generate_plot_for_discriminantFunc(trainingset_class1,trainingset_class2,Points_X1_X2,Root4,Root3,Points_X1_X3,Root1,Root2)

#parzen window calculate for a single feature

f_class1, random_points_class1, mean_parzen_1, covariance_parzen_1 = quad.parzen_window_calculate(trainingset_class1,0.3)
print("This is expected mean and covariance of class1 using parzen window", mean_parzen_1,covariance_parzen_1)
f_class2, random_points_class2, mean_parzen_2, covariance_parzen_2 = quad.parzen_window_calculate(trainingset_class2,0.3)
print("This is expected mean and covariance of class2 using parzen window", mean_parzen_2,covariance_parzen_2)

plot.generate_plot_for_parzen_class1(random_points_class1,f_class1)
plot.generate_plot_for_parzen_class2(random_points_class2,f_class2)

# 5-fold cross validation for testing

tru_positive_x1, tru_negative_x1 = quad.tenfold_X1(trainingset_class1,ML_A,ML_B,ML_C)
print ("this is true positive and negative for X1", tru_positive_x1,tru_negative_x1)
tru_positive_x2, tru_negative_x2 = quad.tenfold_X2(trainingset_class2,ML_A,ML_B,ML_C)
print("this is true positive and negative for X2", tru_positive_x2,tru_negative_x2)
print("Accuracy using k-Fold before diagonalization of points using estimated ML parameters:", quad.accuracy(tru_positive_x1,tru_negative_x1,tru_positive_x2,tru_negative_x2))

# quadtratic classifier for diagonalized points
mean_V1,cov_V1,V1 = gn.generation_of_V1()
mean_V2,cov_V2,V2 = gn.generation_of_V2()

# estimated mean for ML
est_MLmean_class1_diag = quad.estimateMean_for_ML(V1,class1_points)
est_MLmean_class2_diag = quad.estimateMean_for_ML(V2,class2_points)
print("This is estimated(ML) Mean for class1 and class2 after diagonalization:", est_MLmean_class1_diag,est_MLmean_class2_diag)

# estimated covariance for ML
est_MLcovariance_class1_diag = quad.estimateCovariance_forML(V1,class1_points,est_MLmean_class1_diag)
est_MLcovariance_class2_diag = quad.estimateCovariance_forML(V2,class2_points,est_MLmean_class2_diag)
print("This is estimated(ML) covariances for class1 and class2 after diagonalization:", est_MLcovariance_class1_diag,est_MLcovariance_class2_diag)

# estimated mean for Bayesian
est_BLmean_class1_diag = quad.estimateMean_for_Bayesian(V1,class1_points,est_MLcovariance_class1_diag)
est_BLmean_class2_diag = quad.estimateMean_for_Bayesian(V2,class2_points,est_MLcovariance_class2_diag)
print ("this is estimated(BL) mean for class1 and class2 after diagonalization",est_BLmean_class1, est_BLmean_class2)

# find quadtratic classifiers domain(A,B,C) after diagonalization
V1_A,V2_B,V3_C = quad.afterDiag_a_b_c(est_MLmean_class1_diag,est_MLcovariance_class1_diag,est_MLmean_class2_diag,est_MLcovariance_class2_diag)

#plotting discriminant function on V1-V2 and V1-V3 domain for diagonalized points
Root5,Root6,Points_V1_V3 = quad.discriminant_function_X1X3(V1_A,V2_B,V3_C)
Root7,Root8,Points_V1_V2 = quad.discriminant_function_X1X2(V1_A,V2_B,V3_C)

plot.generate_plot_for_discriminantFunc_V(V1,V2,Points_V1_V2,Root7,Root8,Points_V1_V3,Root5,Root6)

#parzen window calculate for a single feature of diagonalized points

f_V1, random_points_V1, mean_parzen_V1, covariance_parzen_V1 = quad.parzen_window_calculate(V1,0.4)
print("This is expected mean and covariance of V1 using parzen window", mean_parzen_V1,covariance_parzen_V1)
f_V2, random_points_V2, mean_parzen_V2, covariance_parzen_V2 = quad.parzen_window_calculate(V2,0.5)
print("This is expected mean and covariance of V2 using parzen window", mean_parzen_V2,covariance_parzen_V2)

#plot.generate_plot_for_parzen_class1(random_points_V1,f_V1)
#plot.generate_plot_for_parzen_class2(random_points_V2,f_V2)

# 5-fold cross validation for testing of V1 and V2

tru_positive_V1, tru_negative_V1 = quad.tenfold_V1(V1,V1_A,V2_B,V3_C)
print ("this is true positive and negative for V1", tru_positive_V1,tru_negative_V1)
tru_positive_V2, tru_negative_V2 = quad.tenfold_V2(V2,V1_A,V2_B,V3_C)
print("this is true positive and negative for V2", tru_positive_V2,tru_negative_V2)
print("Accuracy using k-Fold after diagonalization of points using estimated ML parameters:", quad.accuracy(tru_positive_V1,tru_negative_V1,tru_positive_V2,tru_negative_V2))

# ---------------KNN classifier------------#
'''print("KNN for non-diagonalized points")
diag_points_total = np.append(V1,V2,axis=0)
np.savetxt("diag_data.csv",diag_points_total,delimiter=",")
knn.main_cal('data_mod.csv')
print("KNN for diagonalized points")'''


#-------------Ho-Kashyap----------------#
np.savetxt("diag_data_V1.csv",V1,delimiter=",")
np.savetxt("diag_data_V2.csv",V2,delimiter=",")

#hp.create_Y(trainingset_class1,trainingset_class2)

#-----------Fisher Discriminant----------#

within_class_distance = fd.Sw(est_MLcovariance_class1,est_MLcovariance_class2)
W = fd.W(within_class_distance,est_MLmean_class1,est_MLmean_class2)
fisher_Mean_class1 = fd.fisherMean(W,est_MLmean_class1)
fisher_Mean_class2 = fd.fisherMean(W,est_MLmean_class2)
fisher_Variance_class1 = fd.fisherMean(W,est_MLcovariance_class1)
fisher_Variance_class2 = fd.fisherMean(W,est_MLcovariance_class2)
fisherPoints_class1 = fd.fisherPoint(W,trainingset_class1)
fisherPoints_class2 = fd.fisherPoint(W,trainingset_class2)
#classPoints1 = fd.fisherPlotPoints(W,fisherPoints_class1)
#classPoints2 = fd.fisherPlotPoints(W,fisherPoints_class2)
fd.fisherPlot(fisherPoints_class1,fisherPoints_class2,'before')

#------fishers after diag---------#

within_class_distance_V = fd.Sw(est_MLcovariance_class1_diag,est_MLcovariance_class2_diag)
W_V = fd.W(within_class_distance_V,est_MLmean_class1_diag,est_MLmean_class2_diag)
fisher_Mean_class1_diag = fd.fisherMean(W_V,est_MLmean_class1_diag)
fisher_Mean_class2_diag = fd.fisherMean(W_V,est_MLmean_class2_diag)
fisher_Variance_class1_diag = fd.fisherMean(W_V,est_MLcovariance_class1_diag)
fisher_Variance_class2_diag = fd.fisherMean(W_V,est_MLcovariance_class2_diag)
fisherPoints_class1_diag = fd.fisherPoint(W_V,V1)
fisherPoints_class2_diag = fd.fisherPoint(W_V,V2)
#classPoints1 = fd.fisherPlotPoints(W,fisherPoints_class1)
#classPoints2 = fd.fisherPlotPoints(W,fisherPoints_class2)
fd.fisherPlot(fisherPoints_class1_diag,fisherPoints_class2_diag,'after')