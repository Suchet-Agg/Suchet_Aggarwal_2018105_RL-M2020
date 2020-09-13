## Suchet Aggarwal
## 2018105

import numpy as np
import random
import matplotlib.pyplot as plt 
from math import sqrt,log,exp

num_of_time_steps = 1000

def bandit_sim(epsilon,stddev = 1):
	## Fixing GLobal Variables
	num_of_runs = 2000
	num_of_time_steps = 1000
	num_of_arms = 10
	mean = 0
	Overall_Estimates = []
	Overall_OPT = []
	Overall_Error = []

	## Flag for the case when epsilon is converging acc to eq 2.7
	f = False
	if epsilon == -1:
		f = True

	## Average Over num_of_runs iterations
	for j in range(num_of_runs):

		## Draw Random Reward Means for each arm
		q_star = np.random.normal(mean, stddev, num_of_arms)
		Q = [0]*num_of_arms
		N = [0]*num_of_arms
		Rewards = []
		OPT = []
		Error = []
		if f:
			epsilon = -1
		print("Epsilon :",epsilon, " On Run:", j+1)
		for i in range(num_of_time_steps):

			## Tiem Steps
			A = -1

			## if flag is set, then change epsilon accordingly
			if f:
				epsilon = 50/(i+1)
			probability = max(0,1-epsilon)

			if random.random() < probability:
				## Exploit
				A = np.argmax(Q)
			else:
				## Explore
				A = random.randint(0,num_of_arms-1)

			R = np.random.normal(q_star[A],1,1)[0]
			OPT.append(1 if A==np.argmax(q_star) else 0)
			N[A] += 1

			## Sample Mean Avg
			Q[A] = Q[A] + (1/N[A])*(R-Q[A])
			Error_in_arms = []
			for i in range(num_of_arms):
				Error_in_arms.append(abs(q_star[i]-Q[i]))
			Error.append(Error_in_arms)
			Rewards.append(R)

		Overall_Estimates.append(Rewards)
		Overall_OPT.append(OPT)
		Overall_Error.append(Error)

	Overall_Estimates = np.array(Overall_Estimates)
	Overall_OPT = np.array(Overall_OPT)
	Overall_Error = np.array(Overall_Error)
	Overall_OPT = np.sum(Overall_OPT, axis = 0)
	Overall_OPT = Overall_OPT/20
	Overall_Error = np.sum(Overall_Error, axis = 0)
	Overall_Error = Overall_Error/2000
	results = np.mean(Overall_Estimates,axis=0)
	return [results,Overall_OPT,Overall_Error]

def ucb(c):
	## Fixing GLobal Variables
	num_of_runs = 2000
	num_of_time_steps = 1000
	num_of_arms = 10
	mean = 0
	stddev = 1
	Overall_Estimates = []

	## Average Over num_of_runs iterations
	for i in range(num_of_runs):
		print("On Run:", i+1)
		q_star = np.random.normal(mean, stddev, num_of_arms)
		Q = [5]*num_of_arms
		N = [0]*num_of_arms
		Rewards = []
		for i in range(num_of_time_steps):
			# A = -1

			## q_temp stores q_t + c*(sqrt(ln(t)/N[t]))
			q_temp = list(Q)
			for x in range(len(q_temp)):

				## incase N[x] is 0, set value to 1e9, practically meaning infinite
				if N[x] == 0:
					q_temp[x] = 1e9
				else:
					q_temp[x] += c*sqrt(log(i+1)/N[x])

			## Take argmax of the values of q_temp
			A = np.argmax(q_temp)
			R = np.random.normal(q_star[A],1,1)[0]
			N[A] += 1
			Q[A] = Q[A] + (1/N[A])*(R-Q[A])
			Rewards.append(R)
		Overall_Estimates.append(Rewards)

	Overall_Estimates = np.array(Overall_Estimates)
	results = np.mean(Overall_Estimates,axis=0)
	return results

def bandit_sim_non_stationary(epsilon=0.01, alpha = -1):
	## Fixing GLobal Variables
	num_of_runs = 2000
	num_of_time_steps = 10000
	num_of_arms = 10
	mean = 0
	stddev = 1
	mean_incr = 0
	std_incr = 0.01
	Overall_Estimates = []
	Overall_OPT = []

	## Flag for the case when alpha (constant step size) is not used
	f = False
	if alpha == -1:
		f = True

	## Average Over num_of_runs iterations
	for i in range(num_of_runs):
		print("On Run:", i+1)

		## Draw Random Reward Means for each arm
		q_star_intial = np.random.normal(mean, stddev)

		## q_star for each arm a start out equal
		q_star = [q_star_intial]*num_of_arms
		Q = [0]*num_of_arms
		N = [0]*num_of_arms
		OPT = []
		Rewards = []

		for i in range(num_of_time_steps):
			A = -1
			probability = 1-epsilon
			if random.random() < probability:
				## Exploit
				A = np.argmax(Q)
			else:
				## Explore
				A = random.randint(0,num_of_arms-1)

			R = np.random.normal(q_star[A],1,1)[0]
			N[A] += 1

			## if flag is set, then change use sample average, else constant step size
			if not f:
				Q[A] = Q[A] + (alpha)*(R-Q[A])
			else:
				Q[A] = Q[A] + (1/N[A])*(R-Q[A])

			Rewards.append(R)
			OPT.append(1 if A==np.argmax(q_star) else 0)

			## Update Q_star values
			for i in range(num_of_arms):
				q_star[i] = q_star[i] + np.random.normal(mean_incr,std_incr)
		Overall_Estimates.append(Rewards)
		Overall_OPT.append(OPT)

	Overall_Estimates = np.array(Overall_Estimates)
	results = np.mean(Overall_Estimates,axis=0)
	Overall_OPT = np.array(Overall_OPT)
	Overall_OPT = np.sum(Overall_OPT, axis = 0)
	Overall_OPT = Overall_OPT/20
	return [results,Overall_OPT]

def pi(H,a):
	den = 0
	for i in H:
		den += exp(i)
	return exp(H[a])/den

def gradient_bandit(alpha, baseline=True):
	## Fixing GLobal Variables
	num_of_runs = 2000
	num_of_time_steps = 1000
	num_of_arms = 10
	mean = 4
	stddev = 1
	Overall_Estimates = []
	Overall_OPT = []
	Arms = [i for i in range(num_of_arms)]

	## Average Over num_of_runs iterations
	for j in range(num_of_runs):
		print("On Run", j+1)

		## Draw Random Reward Means for each arm
		q_star = np.random.normal(mean, stddev, num_of_arms)
		H = [0]*num_of_arms
		R_t = 0
		Rewards = []
		OPT = []
		for i in range(num_of_time_steps):

			## Gives the stochastic policy for choosing the best arm
			pi_t = [pi(H,i) for i in range(num_of_arms)]
			A = random.choices(Arms, weights=pi_t, k=1)[0]
			R = np.random.normal(q_star[A],1,1)[0]

			## Update the baseline
			R_t = (i/(i+1))*R_t + (1/(i+1))*R

			if not baseline:
				R_t = 0

			## Update Function H(t) for each arm
			for i in range(A):
				H[i] = H[i] - alpha*(R-R_t)*pi(H,i)

			H[A] = H[A] + alpha*(R-R_t)*(1-pi(H,A))

			for i in range(A+1,num_of_arms):
				H[i] = H[i] - alpha*(R-R_t)*pi(H,i)

			Rewards.append(R)
			OPT.append(1 if A==np.argmax(q_star) else 0)
		Overall_Estimates.append(Rewards)
		Overall_OPT.append(OPT)

	Overall_Estimates = np.array(Overall_Estimates)
	Overall_OPT = np.array(Overall_OPT)

	Overall_OPT = np.sum(Overall_OPT, axis = 0)
	Overall_OPT = Overall_OPT/20

	results = np.mean(Overall_Estimates,axis=0)
	return [results,Overall_OPT]

def Question1(choice):
	if choice == 1:
		## Avg Rewards
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(-1)[0], label = 'e = Varying') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0)[0], label = 'e = 0') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.1)[0], label = 'e = 0.1') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.01)[0], label = 'e = 0.01') 
		plt.ylabel('Avg Reward') 

	elif choice == 2:
		## Optimal Action
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(-1)[1], label = 'e = Varying') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0)[1], label = 'e = 0') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.1)[1], label = 'e = 0.1') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.01)[1], label = 'e = 0.01') 
		plt.ylabel('% Optimal Action')  

	else:
		## Relative Error
		a = bandit_sim(0.1)[2]
		plt.title('Epsilon = 0.1')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		a = bandit_sim(0.01)[2]
		plt.title('Epsilon = 0.01')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		a = bandit_sim(0)[2]
		plt.title('Epsilon = 0')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		a = bandit_sim(-1)[2]
		plt.title('Epsilon = Varying')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		plt.ylabel('Absolute Error')  

def Question2(choice):
	if choice==1:
		## Avg Reward
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(-1,2)[0], label = 'e = Varying') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0,2)[0], label = 'e = 0') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.1,2)[0], label = 'e = 0.1') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.01,2)[0], label = 'e = 0.01') 
		plt.ylabel('Avg Reward') 

	elif choice ==2:
		## Optimal Action
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(-1,2)[1], label = 'e = Varying') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0,2)[1], label = 'e = 0') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.1,2)[1], label = 'e = 0.1') 
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.01,2)[1], label = 'e = 0.01') 
		plt.ylabel('% Optimal Action')  

	else:
		## Relative Error
		a = bandit_sim(0.1,2)[2]
		plt.title('Epsilon = 0.1')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		a = bandit_sim(0.01,2)[2]
		plt.title('Epsilon = 0.01')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		a = bandit_sim(0,2)[2]
		plt.title('Epsilon = 0')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		a = bandit_sim(-1,2)[2]
		plt.title('Epsilon = Varying')
		for i in range(10):
			plt.plot([i for i in range(1,num_of_time_steps+1)], a[:,i], label = 'Arm = '+str(i))	
		plt.ylabel('Absolute Error')  

def Question5(choice):
	if choice == 1:
		## Avg Reward
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim_non_stationary(0.1,-1)[0], label = 'Sample Mean Averages')
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim_non_stationary(0.1,0.1)[0], label = 'Constant Step size: aplha = 0.1')	
		plt.ylabel('Avg Reward') 

	else:
		## Optimal Action
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim_non_stationary(0.1,-1)[0], label = 'Sample Mean Averages')
		plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim_non_stationary(0.1,0.1)[0], label = 'Constant Step size: aplha = 0.1')	
		plt.ylabel('% Optimal Action')  

def Question6():
	plt.plot([i for i in range(1,num_of_time_steps+1)], bandit_sim(0.1)[0], label = 'e = 0.1') 
	plt.plot([i for i in range(1,num_of_time_steps+1)], ucb(1), label = 'UCB: c = 1') 
	plt.plot([i for i in range(1,num_of_time_steps+1)], ucb(2), label = 'UCB: c = 2') 
	plt.plot([i for i in range(1,num_of_time_steps+1)], ucb(4), label = 'UCB: c = 4') 
	plt.ylabel('Avg Reward') 

def Question7():
	plt.plot([i for i in range(1,num_of_time_steps+1)], gradient_bandit(0.1,True)[1], label = 'alpha = 0.1 with Baseline') 
	plt.plot([i for i in range(1,num_of_time_steps+1)], gradient_bandit(0.1,False)[1], label = 'alpha = 0.1 without Baseline') 
	plt.plot([i for i in range(1,num_of_time_steps+1)], gradient_bandit(0.4,True)[1], label = 'alpha = 0.4 with Baseline') 
	plt.plot([i for i in range(1,num_of_time_steps+1)], gradient_bandit(0.4,False)[1], label = 'alpha = 0.4 without Baseline')
	plt.ylabel('% Optimal Action')  

## Call Your Function Here
Question2(3)
plt.xlabel('Steps') 
plt.legend()
plt.show() 