import matplotlib.pyplot as plt

def slicing_points(X1, X2):
    X1_0_1 = X1[:,[2,4]] # X1 - X2
    X1_0_2 = X1[:,[2,4]] # X1 - X3

    X2_0_1 = X2[:,[2,4]]
    X2_0_2 = X2[:,[2,4]]

    return X1_0_1,X1_0_2,X2_0_1,X2_0_2

#plotting 200 training points before diagonalization
def generate_plot_for_X(plotX1,plotX2):
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = slicing_points(plotX1, plotX2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')

    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.title("Before Diagonalization(X1 - X2) domain")
    plt.show()
    # To plot X1_X3 domain of X
    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')

    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("Before Diagonalization(X1 - X3) domain")
    plt.show()

def generate_plot_for_HoKashyap(class1,class2,plot_x,plot_y, string):
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = slicing_points(class1, class2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    plt.scatter(plot_x, plot_y, c='green')

    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.title("Discriminant Function for HoKashyap" + str(string) + "diagonalization")
    plt.show()

def generate_plot_for_discriminantFunc(plotX1,plotX2,Points_X1_X2,Root4,Root3,Points_X1_X3,Root1,Root2):
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = slicing_points(plotX1, plotX2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    #plt.scatter(Points_X1_X2[:], Root4[:], c='green')
    plt.scatter(Points_X1_X2[:], Root3[:], c='green')

    #plt.text( 2, 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.title("Discriminant function before diagonalized points(X1 - X2) domain")
    plt.show()


    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')
    #plt.scatter(Points_X1_X3[:], Root1[:], c='green')
    plt.scatter(Points_X1_X3[:], Root2[:], c='green')

    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("Discriminant function before diagonalized points(X1 - X3) domain")
    plt.show()

def generate_plot_for_discriminantFunc_V(plotX1,plotX2,Points_X1_X2,Root4,Root3,Points_X1_X3,Root1,Root2):
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = slicing_points(plotX1, plotX2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    #plt.scatter(Points_X1_X2[:], Root4[:], c='green')
    plt.scatter(Points_X1_X2[:], Root3[:], c='green')

    #plt.text( 2, 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.title("Discriminant function after diagonalized points(V1 - V2) domain")
    plt.show()


    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')
    #plt.scatter(Points_X1_X3[:], Root1[:], c='green')
    plt.scatter(Points_X1_X3[:], Root2[:], c='green')

    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("Discriminant function after diagonalized points(V1 - V3) domain")
    plt.show()

def generate_plot_for_parzen_class1(random_points,f_class):
    plt.title("Final learned distribution(number of MA founds) in class 1")
    plt.xlabel("random points for class 1")
    plt.ylabel("estimated contribution for each random point")
    plt.scatter(random_points, f_class)
    plt.show()

def generate_plot_for_parzen_class2(random_points,f_class):
    plt.title("Final learned distribution(number of MA founds) in class 2")
    plt.xlabel("random points for class 2")
    plt.ylabel("estimated contribution for each random point")
    plt.scatter(random_points, f_class)
    plt.show()