import numpy as np
import matplotlib.pyplot as plt
import quadtratic as quad
import t3utility as t3

# class 1 points generation from dataset
filepath1 = t3.currentFilePath('training_class1.csv')
trainingset_class1 = np.genfromtxt(filepath1, dtype=float, delimiter=',')
trainingset_class1 = t3.getSelectedColumns(trainingset_class1, (2, 4, 6, 11, 15, 17))
np.savetxt("class1.csv",trainingset_class1,delimiter=",")

class1_points,class1_col = trainingset_class1.shape

# class 2 points generation from dataset
filepath2 = t3.currentFilePath('training_class2.csv')
trainingset_class2 = np.genfromtxt(filepath2, dtype=float, delimiter=',')
trainingset_class2 = t3.getSelectedColumns(trainingset_class2, (2, 4, 6, 11, 15, 17))
np.savetxt("class2.csv",trainingset_class2,delimiter=",")

class2_points,class2_col = trainingset_class2.shape

# estimated mean for ML
est_MLmean_class1 = quad.estimateMean_for_ML(trainingset_class1,class1_points)
est_MLmean_class2 = quad.estimateMean_for_ML(trainingset_class2,class2_points)
print("This is estimated(ML) Mean for class1 and class2:", est_MLmean_class1,est_MLmean_class2)

# estimated covariance for ML
est_MLcovariance_class1 = quad.estimateCovariance_forML(trainingset_class1,class1_points,est_MLmean_class1)
est_MLcovariance_class2 = quad.estimateCovariance_forML(trainingset_class2,class2_points,est_MLmean_class2)
print("This is estimated(ML) covariances for class1 and class2:", est_MLcovariance_class1,est_MLcovariance_class2)


X1 = trainingset_class1
X2 = trainingset_class2
sigma1 = est_MLcovariance_class1
sigma2 = est_MLcovariance_class2
mean_X1 = est_MLmean_class1
mean2_X2 = est_MLmean_class2

# generation of Y1 and Y2
def generation_of_Y1():
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    Px1_transpose = np.transpose(Px1)

    y1 = np.dot(Px1_transpose,X1.transpose())
    y1 = y1.transpose()

    # mean of Y1
    mean_of_y1 = np.dot(Px1_transpose,mean_X1)
    print("Mean of Y1:", mean_of_y1)

    # covariance of Y1
    covariance_of_Y1 = np.dot(np.dot(Px1_transpose,sigma1),Px1)
    print("Covariance of Y1:", covariance_of_Y1)

    # eigenvalues and eigenvectors of Y1

    eigenvalue_Y1, eigenvector_Y1 = np.linalg.eig(covariance_of_Y1)
    print("eigen values of  Y1:", eigenvalue_Y1)
    print("eigen vectors of Y1:", eigenvector_Y1)
    return y1


def generation_of_Y2():

    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    Px1_transpose = np.transpose(Px1)

    y2 = np.dot(Px1_transpose, X2.transpose())
    y2 = y2.transpose()

    # mean of Y2
    mean_of_y2 = np.dot(Px1_transpose,mean2_X2)
    print("Mean of Y2:", mean_of_y2)

    covariance_of_Y2 = np.dot(np.dot(Px1_transpose,sigma2),Px1)
    print("Covariance of Y2:", covariance_of_Y2)

    # eigenvalues and eigenvectors of Y2

    eigenvalue_Y2, eigenvector_Y2 = np.linalg.eig(covariance_of_Y2)
    print("eigen values of  Y2:", eigenvalue_Y2)
    print("eigen vectors of Y2:", eigenvector_Y2)
    return y2

Y1 = generation_of_Y1()
Y2 = generation_of_Y2()

# generation of Z1 and Z2
def generation_of_Z1():
    lambda_def, Px1 = (np.linalg.eig(sigma1)) # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    z1 = np.dot(np.dot(tempZ_1,Px1_transpose),X1.transpose())
    z1 = z1.transpose()

    # mean of Z1

    mean_of_Z1 = np.dot(np.dot(tempZ_1,Px1_transpose),mean_X1)
    print("Mean of Z1:", mean_of_Z1)

    # covariance of Z1
    covariance_of_Z1 = np.identity(6)
    print ("This is covariance of Z1:", covariance_of_Z1)

    # eigenvalues and eigenvectors of Z1

    eigenvalue_Z1, eigenvector_Z1 = np.linalg.eig(covariance_of_Z1)
    print("eigen values of  Z1:", eigenvalue_Z1)
    print("eigen vectors of Z1:", eigenvector_Z1)
    return  z1


def generation_of_Z2():
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    z2 = np.dot(np.dot(tempZ_1, Px1_transpose), X2.transpose())
    z2 = z2.transpose()

    # mean of Z2
    mean_Z2 = np.dot(np.dot(tempZ_1,Px1_transpose),mean2_X2)
    print("Mean of Z2:", mean_Z2)

    # covariance of Z2
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1,Px1_transpose),sigma2),Px1), tempZ_1)
    print("covariance of Z2:", covariance_of_Z2)

    # eigenvalues and eigenvectors of Z2
    eigenvalue_Z2, eigenvector_Z2 = np.linalg.eig(covariance_of_Z2)
    print("eigen values of Z2:", eigenvalue_Z2)
    print("eigen vectors of Z2:", eigenvector_Z2)
    return z2

Z1 = generation_of_Z1()
Z2 = generation_of_Z2()
# generation of V1 and V2

def generation_of_V1():
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    # covariance of Z2
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1, Px1_transpose), sigma2), Px1), tempZ_1)
    eigenvalue,Pz2 = (np.linalg.eig(covariance_of_Z2))
    Pz2_transpose = Pz2.transpose()

    v1 = np.dot(Pz2_transpose,Z1.transpose())
    v1 = v1.transpose()

    # mean and covariance of V1
    mean_of_Z1 = np.dot(np.dot(tempZ_1, Px1_transpose), mean_X1)
    mean_V1 = np.dot(Pz2_transpose,mean_of_Z1)
    print ("The mean of V1:", mean_V1)

    I = np.identity(6)

    covariance_of_V1 = np.dot(np.dot(Pz2_transpose,I),Pz2)
    print ("The covariance of V1:", covariance_of_V1)
    return mean_V1,covariance_of_V1,v1


def generation_of_V2():
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    # covariance of Z2
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1, Px1_transpose), sigma2), Px1), tempZ_1)
    som, Pz2 = (np.linalg.eig(covariance_of_Z2))
    Pz2_transpose = Pz2.transpose()

    v2 = np.dot(Pz2_transpose, Z2.transpose())
    v2 = v2.transpose()

    # mean and covariance of V1
    mean_of_Z2 = np.dot(np.dot(tempZ_1, Px1_transpose), mean2_X2)
    mean_V2 = np.dot(Pz2_transpose, mean_of_Z2)
    print ("The Mean of V2:", mean_V2)

    covariance_of_V2 = np.dot(np.dot(Pz2_transpose, covariance_of_Z2), Pz2)
    print ("The covariance of V2:", covariance_of_V2)

    return mean_V2,covariance_of_V2,v2

m_v1,c_v1,V1 = generation_of_V1()
m_v2,c_v2,V2 = generation_of_V2()

def slicing_points(X1, X2):
    X1_0_1 = X1[:,[0,1]] # X1 - X2
    X1_0_2 = X1[:,[0,2]] # X1 - X3

    X2_0_1 = X2[:,[0,1]]
    X2_0_2 = X2[:,[0,2]]

    return X1_0_1,X1_0_2,X2_0_1,X2_0_2


def generate_plot():
    X1_0_1,X1_0_2,X2_0_1,X2_0_2 = slicing_points(X1,X2)

    # To plot X1_X2 domain of X
    a = np.amax(X1)
    b = np.amin(X1)
    c = np.amax(X2)
    d = np.amin(X2)


    plt.axis([d-5,c+5,b,a])

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.axis([d, c, b, a])
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    plt.text(d+2,a-2,'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("X1 - X2 domain")
    plt.show()

    plt.axis([d-3, c+3, b-2, a])

    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')
    plt.text(d + 2, a - 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("X1 - X3 domain")
    plt.show()

    a = np.amax(Y1)
    b = np.amin(Y1)
    c = np.amax(Y2)
    d = np.amin(Y2)

    # To plot Y1_Y2 domain of Y
    Y1_0_1, Y1_0_2, Y2_0_1, Y2_0_2 = slicing_points(Y1, Y2)

    plt.axis([d - 3, c + 8, b - 6, a + 1])

    plt.scatter(Y1_0_1[:, [0]], Y1_0_1[:, [1]], c='red')
    plt.scatter(Y2_0_1[:, [0]], Y2_0_1[:, [1]], c='blue')
    plt.text(d + 2, a - 2, 'red= first, blue = second')
    plt.xlabel("Y1")
    plt.ylabel("Y2")
    plt.title("Y1 - Y2 domain")
    plt.show()

    plt.axis([d - 3, c + 9, b -2 , a])

    plt.scatter(Y1_0_2[:, 0], Y1_0_2[:, 1], c='red')
    plt.scatter(Y2_0_2[:, 0], Y2_0_2[:, 1], c='blue')
    plt.text(d + 2, a - 2, 'red= first, blue = second')
    plt.xlabel("Y1")
    plt.ylabel("Y3")
    plt.title("Y1 - Y3 domain")
    plt.show()

    a = np.amax(Z1)
    b = np.amin(Z1)
    c = np.amax(Z2)
    d = np.amin(Z2)

    Z1_0_1, Z1_0_2, Z2_0_1, Z2_0_2 = slicing_points(Z1, Z2)

    # To plot Z1_Z2 domain of Z

    plt.axis([d-3, c+3, b-5, a+2])

    plt.scatter(Z1_0_1[:, [0]], Z1_0_1[:, [1]], c='red')
    plt.scatter(Z2_0_1[:, [0]], Z2_0_1[:, [1]], c='green')
    plt.text(d + 0.1, a - 0.1, 'red= first, green = second')
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.title("Z1 - Z2 domain")
    plt.show()

    plt.axis([d , c , b-3, a])

    plt.scatter(Z1_0_2[:, 0], Z1_0_2[:, 1], c='red')
    plt.scatter(Z2_0_2[:, 0], Z2_0_2[:, 1], c='green')
    plt.text(d + 0.6, a - 0.6, 'red= first, green = second')
    plt.xlabel("Z1")
    plt.ylabel("Z3")
    plt.title("Z1 - Z3 domain")
    plt.show()

    a = np.amax(V1)
    b = np.amin(V1)
    c = np.amax(V2)
    d = np.amin(V2)

    V1_0_1, V1_0_2, V2_0_1, V2_0_2 = slicing_points(V1, V2)

    # To plot V1_V2 domain of V

    plt.axis([d-3 , c+0.5 , b-2, a])

    plt.scatter(V1_0_1[:, [0]], V1_0_1[:, [1]], c='red')
    plt.scatter(V2_0_1[:, [0]], V2_0_1[:, [1]], c='green')
    plt.text(d + 2, a - 0.5, 'red= first, green = second')
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.title("V1 - V2 domain")
    plt.show()

    plt.axis([d-2 , c+2 , b-3, a+3])

    plt.scatter(V1_0_2[:, 0], V1_0_2[:, 1], c='red')
    plt.scatter(V2_0_2[:, 0], V2_0_2[:, 1], c='green')
    plt.text(d + 0.2, a+2, 'red= first, green = second')
    plt.xlabel("V1")
    plt.ylabel("V3")
    plt.title("V1 - V3 domain")
    plt.show()


def generate_POverall():
    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    # covariance of Z2

    sigma2 = generate_sigma2()
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1, Px1_transpose), sigma2), Px1), tempZ_1)
    eigenvalue,Pz2 = (np.linalg.eig(covariance_of_Z2))
    Pz2_transpose = Pz2.transpose()

    P_overall = np.dot(np.dot(Pz2_transpose,tempZ_1),Px1_transpose)

    return P_overall
