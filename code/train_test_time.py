import matplotlib.pyplot as plt
import matplotlib
import numpy as np
def draw(x, Y, leg):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # matplotlib.
    for i in range(len(Y)):
        res = ax.plot(x, Y[i], '-o', label=leg[i])
    # plt.xlabel('# of Ratings')
    plt.xlabel('# of Ratings(log base 10 scale)')
    plt.ylabel('Traing/Test Time in seconds')
    plt.title('Training/Test Time vs # of Ratings')
    plt.legend()

Y_train =  [19.71, 143.09, 587.56, 1192.90]
Y_testing_time_on_test =  [0.16, 0.33, 6.91 , 13.59]
Y_testing_time_on_train =  [0.03,1.55 ,31.93 ,62.50 ]

X = [i for i in [100000, 1000000, 10000000, 20000000]]
X_logbase_10 = np.log10([i for i in [100000, 1000000, 10000000, 20000000]])
Y = [Y_train, Y_testing_time_on_train, Y_testing_time_on_test]
leg = ['Training Time', 'Testing time on Training Data', 'Testing time on Test Data']
# draw(X, Y, leg)
draw(X_logbase_10, Y, leg)
plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# for i in range(10):
#     ov= []
#     rns = []
#     w = i+1
#     for j in range(50):
#         u = j+1
#         sc = m[(u, w)]
#         if len(sc) > 0:
#             ov.append(u)
#             rns.append(np.mean(sc))

#     ax2.plot(ov, rns)
#     overs = np.arange(1, 51, 1)
#     runs = params[i] * (1.0 - np.exp((-params[10] * overs)/params[i]))
#     ax1.plot(overs, runs)
#     ax1.text(50, params[i] * (1.0 - np.exp((-params[10] * 50)/params[i])), r'w = {}'.format(i+1), fontsize = 8 )

# ax1.set_xlabel('Overs Remaining')
# ax1.set_ylabel('Expected Runs')
# ax1.set_title('Fitted Curve')
# ax2.set_xlabel('Overs Remaining')
# ax2.set_ylabel('Average Runs')
# ax2.set_title('Observed Data')
