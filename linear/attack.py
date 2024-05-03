import numpy as np
import random
from scipy.optimize import linprog, fmin


class Attack:

    def __init__(self, attack_type):
        self.attack_type = attack_type

    def attack(self, X_test, y_test_all, sgd_clf, problems, n_selected, n_attk_samples, lb, ub):
        n_features = X_test.shape[1]*X_test.shape[2]
        selected = random.sample(range(len(problems)), n_selected)
        clf_attack = [problems[i] for i in selected] 
        clf_protect = [problems[i] for i in range(len(problems)) if i not in selected]
        print('attacked: ', clf_attack)
        print('protected: ', clf_protect)

        #ranomly select a slice of n_attk_samples samples to attack
        n_start = random.randint(0, X_test.shape[0]-n_attk_samples)

        if self.attack_type == 'l1':
            print('[INFO:] '+self.attack_type+' attacking ...')
            #The coefficients of the linear objective function f(delta_xi, ti) to be minimized
            # coef of delta_xi
            c = [0]*n_features
            # coef of ti
            c.extend([1]*n_features)

            #The 2D inequality constraint matrix for variables delta_xi and ti
            A=[]

            #The inequality constraint vector
            b=[]

            for i in range(n_features):
                a1= [0]*n_features
                a1[i] = 1
                a1.extend([0]*n_features)
                a1[i+n_features] = -1
                A.append(a1)
                b.append(0)
                a2 = [0]*n_features
                a2[i] = -1
                a2.extend([0]*n_features)
                a2[i+n_features] = -1
                A.append(a2)
                b.append(0)


            adv_x=[]

            for i in range(n_start, n_start+n_attk_samples):
                A_ub = A.copy()
                b_ub = b.copy()
                x = X_test[i].flatten()

                # loop through attacked models
                clf_attack_ind = [problems.index(t) for t in clf_attack]
                for j in clf_attack_ind:
                   y_test = y_test_all[j]
                   y = y_test[i] 
                   w = sgd_clf[j].coef_.flatten()
                   a_ = [y*t for t in w]
                   a_.extend([0]*n_features)
                   b_ = -(sum([t*p for t,p in zip(a_, x)]) + y*sgd_clf[j].intercept_[0])
                   A_ub.append(a_)
                   b_ub.append(b_)

                # loop through protected models
                clf_protect_ind = [problems.index(t) for t in clf_protect]
                for j in clf_protect_ind:
                   y_test = y_test_all[j]
                   y = y_test[i] 
                   w = sgd_clf[j].coef_.flatten()
                   a_ = [-y*t for t in w]
                   a_.extend([0]*n_features)
                   a0_ = [y*t for t in w]
                   b_ = sum([t*p for t,p in zip(a0_, x)]) + y*sgd_clf[j].intercept_[0]
                   A_ub.append(a_)
                   b_ub.append(b_)
                t_bounds=(0,None)
                dx_bounds=(lb, ub) 
                bounds=[dx_bounds]*n_features
                bounds.extend([t_bounds]*n_features)
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
                adv_x.append(x + res.x[0:n_features])

        elif self.attack_type == 'linf':

            print('[INFO:] '+self.attack_type+' attacking ...')
            #The coefficients of the linear objective function f(delta_xi, t) to be minimized
            # coef of delta_xi
            c = [0]*n_features
            # coef of t
            c.extend([1]*1)

            #The 2D inequality constraint matrix for variables delta_xi and ti
            A=[]

            #The inequality constraint vector
            b=[]

            for i in range(n_features):
                a1= [0]*n_features
                a1[i] = 1
                a1.extend([-1])
                A.append(a1)
                b.append(0)
                a2 = [0]*n_features
                a2[i] = -1
                a2.extend([-1])
                A.append(a2)
                b.append(0)

            adv_x=[]
            for i in range(n_start, n_start+n_attk_samples):
                A_ub = A.copy()
                b_ub = b.copy()
                x = X_test[i].flatten()

                # loop through attacked models
                clf_attack_ind = [problems.index(t) for t in clf_attack]
                for j in clf_attack_ind:
                   y_test = y_test_all[j]
                   y = y_test[i] 
                   w = sgd_clf[j].coef_.flatten()
                   a_ = [y*t for t in w]
                   a_.extend([0])
                   b_ = -(sum([t*p for t,p in zip(a_, x)]) + y*sgd_clf[j].intercept_[0])
                   A_ub.append(a_)
                   b_ub.append(b_)

                # loop through protected models
                clf_protect_ind = [problems.index(t) for t in clf_protect]
                for j in clf_protect_ind:
                   y_test = y_test_all[j]
                   y = y_test[i] 
                   w = sgd_clf[j].coef_.flatten()
                   a_ = [-y*t for t in w]
                   a_.extend([0])
                   a0_ = [y*t for t in w]
                   b_ = sum([t*p for t,p in zip(a0_, x)]) + y*sgd_clf[j].intercept_[0]
                   A_ub.append(a_)
                   b_ub.append(b_)
                t_bounds=(0,None)
                dx_bounds=(lb,ub)
                bounds=[dx_bounds]*n_features
                bounds.extend([t_bounds])
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
                adv_x.append(x + res.x[0:n_features])

        elif self.attack_type == 'l2':

            print('[INFO:] '+self.attack_type+' attacking ...')
            from scipy.optimize import Bounds
            C = len(problems)
            bounds = Bounds([0]*C,[np.inf]*C)

            def term_G(lam, y_ind):
                g1 = np.zeros(n_features)
                g2 = np.zeros(n_features)
                clf_attack_ind = [problems.index(t) for t in clf_attack]
                for i in clf_attack_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    g1 = g1 + lam[i]*y*w
                clf_protect_ind = [problems.index(t) for t in clf_protect]
                for i in clf_protect_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    g2 = g2 + lam[i]*y*w
                return g1 - g2

            def term_B(lam, x, y_ind):
                b1 = 0
                b2 = 0
                clf_attack_ind = [problems.index(t) for t in clf_attack]
                for i in clf_attack_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    b1 = b1 + y*lam[i]*sum([t*p for t,p in zip(w, x)]) + y*lam[i]*sgd_clf[i].intercept_[0]
                clf_protect_ind = [problems.index(t) for t in clf_protect]
                for i in clf_protect_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    b2 = b2 + y*lam[i]*sum([t*p for t,p in zip(w, x)]) + y*lam[i]*sgd_clf[i].intercept_[0]
                return b1 - b2

            def obj_func(lam, x, y_ind):
                g1 = np.zeros(n_features)
                g2 = np.zeros(n_features)
                clf_attack_ind = [problems.index(t) for t in clf_attack]
                for i in clf_attack_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    g1 = g1 + lam[i]*y*w
                clf_protect_ind = [problems.index(t) for t in clf_protect]
                for i in clf_protect_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    g2 = g2 + lam[i]*y*w
                t1 = g1 - g2
                t2 = np.transpose(t1)

                b1 = 0
                b2 = 0
                clf_attack_ind = [problems.index(t) for t in clf_attack]
                for i in clf_attack_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    b1 = b1 + y*lam[i]*np.dot(w,x) + y*lam[i]*sgd_clf[i].intercept_[0]
                clf_protect_ind = [problems.index(t) for t in clf_protect]
                for i in clf_protect_ind:
                    y_test = y_test_all[i]
                    y = y_test[y_ind]
                    w = sgd_clf[i].coef_.flatten()
                    b2 = b2 + y*lam[i]*np.dot(w,x) + y*lam[i]*sgd_clf[i].intercept_[0]
                return 0.25*np.dot(t1,t2) - (b1-b2)

            from scipy.optimize import minimize

            adv_x=[]
            for i in range(n_start, n_start+n_attk_samples):
                lam = np.array([0.0]*C)
                x = X_test[i].flatten()
                y_ind = i 
                res = minimize(obj_func, lam, args=(x,y_ind), method='trust-constr', options={'verbose': 0}, bounds=bounds)
                delta_x = -0.5*term_G(res.x,y_ind)
                adv_x.append(x + delta_x)

        return n_start, selected, adv_x
