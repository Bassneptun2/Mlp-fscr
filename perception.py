import numpy as np

daita = np.array([[12.3],[14.2],[11.4],[30.5],[28],[10.7],[13.5],[32.9],[12],[11],[10],[34.2],[32.2],[14.1],[27.9],[13.2],[32.1],[10],[30]])
sol = np.array([0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,0,1,0,1])

data = np.concatenate((daita, np.ones(19).reshape(19, 1)), axis = 1) 
print(sol.shape)
class Neuron:
	def __init__(self, max_iter:int):
		self.max_iter = max_iter
		self.comp = False
		self.weight = None
		self.norm:float = None
		self.fehler = np.zeros(max_iter) 
	
	def neuron(self, x):
		if self.comp: 
			x /= float(self.norm)
		return 1 if np.dot(self.weight, x) > 0 else 0	
		
	def lernfunktion(self, data, sol):
		self.norm = np.max(data, 0)
		data /= self.norm

		self.weight = np.random.rand(data.shape[1]) 
		i = 0
		while i < self.max_iter:
			for x, sols in zip(data, sol):
				fehler = sols - self.neuron(x)
				if fehler != 0:
					self.weight += fehler * x
					self.fehler[i] += 1
			if self.fehler[i] == 0:
				self.comp = True
				break
			i += 1	
		else:
			return "no solution"
neuron = Neuron(10000)
neuron.lernfunktion(data, sol)

test = np.array([13, 10])
print(neuron.neuron(test))
