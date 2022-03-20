import os
import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import matplotlib as mpl
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import Tuple
from torchdiffeq import odeint
from abc import ABC, abstractmethod
from math import pi
from utilities import write_tensor, timeit, ProgressBar


class COCAbstract(ABC):
    # This class handles everything related to coupled oscillator systems
    # Note that this is an abstract class, so it cannot be instantiated

    # Constructor for a Coupled Oscillator Circuit (COC)
    def __init__(self, dateString: str) -> None:
        self.dateString = dateString

        return

    # Building the system from the given file name fileName
    # Note this is the whole read, but the exception checks will be
    # based on the given class which will be actually instantiated
    # TODO: full check with exceptions 
    def build_from_file(self, fileName: str = "default.in") -> None:
        with open(fileName, 'r') as f:
            for line in f.readlines():
                line = line.split(" = ")

                fieldName = line[0].rstrip()
                value = line[1].rstrip()

                if fieldName in ["inputs", "Couplings", "B"]:
                    setattr(self, fieldName, np.fromstring(value, sep=' '))
                elif fieldName in ["oscNum", "tNsamples"]:
                    setattr(self, fieldName, int(value))
                elif fieldName in ["R", "C", "Rc", "Ri", "tBegin", "tEnd", "A", "f"]:
                    setattr(self, fieldName, float(value))
                elif fieldName in ("save", "gpu"):
                    setattr(self, fieldName, bool(int(value)))
                elif fieldName in ("method", "saveFolder"):
                    setattr(self, fieldName, value)
                    
        # Set parameters
        # Note that these can be altered, but for now, it's okay
        self.invNum = 7
        self.refOsc = 0
        self.refNode = 5
        self.targetOscs = range(self.oscNum)
        self.targetNodes = np.full(self.oscNum, 5)
        
        # Check if the class has batchNum attribute and if it doesn't set it to 1
        if not hasattr(self, "batchNum"):
            self.batchNum = 1
            
        # Initialise input amplitudes and frequencies if they are not present 
        if not hasattr(self, "A"):
            self.A = 0.0
        if not hasattr(self, "f"):
            self.f = 0.0
                    
        # Sets the device for the whole simulation
        if hasattr(self, "gpu"):
            if self.gpu:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # If there is any input to the system, save the inputNum
        # for the sake of convencience
        if hasattr(self, "inputs"):
            setattr(self, "inputNum", self.inputs.size)
        else:
            # Set inputNum according to oscNum
            self.inputNum = self.oscNum

        # Reshape the input Couplings and B from row vector to matrix
        if hasattr(self, "Couplings"):
            self.Couplings = self.Couplings.reshape((self.oscNum, self.oscNum))

        if hasattr(self, "B"):
            self.B = self.B.reshape((self.oscNum, self.inputNum))
        else:
            self.B = np.zeros((self.oscNum, self.inputNum))
            
        # Define the integration method
        # By default its dopri8 (Runge-Kutta 4-5, ode45 in MATLAB)
        if not hasattr(self, "method"):
            self.method = "dopri5"
        else:
            if self.method == "implicit":
                self.method = "implicit_adams"

        # Create the tSpace for the calculations
        self.tSpace = torch.tensor(
            np.linspace(self.tBegin, self.tEnd, self.tNsamples),
            dtype=torch.float64, device=self.device
        )
        
        # Modifies the saveFolder to have a unique identifier
        if hasattr(self, "saveFolder"):
            self.saveFolder = self.saveFolder + self.dateString + "/"
            os.mkdir(self.saveFolder)
            os.mkdir(self.saveFolder + "model_cps/")
            
        # Creating the progressbar for the integration
        self.progressBar = ProgressBar(self.tEnd)

        # Create the rest of the system using the inputs
        self.create_P()

        # Plot cutoff
        # Note: the 25 is arbitrary
        self.plotCO = int(25 * self.tNsamples / (self.tEnd*1.0E+9))

        # Set figure identifiers
        self.figRes = 0
        self.figGraph = 1
        self.figPhaseDiff = 2

        return

    # Construction of P permutation matrix
    def create_P(self) -> None:
        # Pi: permutation matrix corresponding to the left cyclical rotation
        # P: permutation matrix in the m-wide main diagonal
        Pi = torch.zeros(self.invNum, self.invNum)
        Pi[0, self.invNum-1] = 1
        for i in range(1, self.invNum):
            Pi[i, i-1] = 1

        self.P = torch.zeros(self.invNum*self.oscNum, self.invNum*self.oscNum,
                             dtype=torch.float64)
        for i in range(self.oscNum):
            self.P[i*self.invNum:(i+1)*self.invNum,
                   i*self.invNum:(i+1)*self.invNum] = Pi

        write_tensor("P.txt", self.P, self.oscNum, self.invNum)
        
        self.P = self.P.to(self.device)

        return

    # Construction of Bprime for inputs
    @abstractmethod
    def create_Bprime(self) -> None:
        pass

    # Construction of Cprime for Couplings
    def create_Cprime(self) -> None:
        if self.type == "full":
            self.create_Cprime_full()
        elif self.type == "NN":
            self.create_Cprime_nearest_neighbour()
        else:
            self.Cprime_from_C()
            
    # This creates a random init value for the ODE
    def create_random_V_init(self) -> None:
        self.V_init = torch.rand(
            (self.oscNum*self.invNum, self.batchNum),
            dtype=torch.float64
        )

        # Scale V_init to [-1, 1] from [0, 1]
        self.V_init = 2*self.V_init - 1

        self.V_init = self.V_init.to(self.device)
        
        return            

    # Abstract method for running the simulation
    @abstractmethod
    def run_simulation(self) -> None:
        pass

    # Nonlinearity for the inverters
    # Note: this scaled the values between -1 and 1 and the 10.0 is empirical
    def invf(self, V: torch.tensor) -> torch.tensor:
        VOut = -torch.tanh(10.0*V)

        return VOut

    # This is the input vector consisting the input signals
    def u(self, t: float) -> torch.tensor:
        u = torch.zeros(self.inputNum, self.batchNum,
                        dtype=torch.float64, device=self.device)
        t = torch.full((self.inputNum, self.batchNum), t.item(),
                       dtype=torch.float64, device=self.device)

        u = self.A * torch.sin(2*pi*self.f*t + self.inputs*pi)

        return u

    # ODEs for the coupled oscillatory system
    def ode_fn(self, t, V) -> torch.tensor:
        self.progressBar.update(t)

        inr_dyn = self.invf(torch.matmul(self.P, V)) - V
        cpl_dyn = torch.matmul(self.Cprime, V)
        ext_dyn = torch.matmul(self.Bprime, self.u(t))

        dV = (inr_dyn/(self.R*self.C)
              + cpl_dyn / (self.Rc*self.C)
              + ext_dyn / (self.Ri*self.C))

        return dV

    # Save the results to file
    def save_results(self, fileName: str = "Results") -> None:
        V = self.V.detach().cpu()

        targets = [i*self.invNum+self.targetNodes[i] for i in self.targetOscs]
        with open(self.saveFolder + fileName, 'w') as f:
            np.savetxt(f, V[-self.plotCO:, targets, 0])

        return

    # This plots the output vectors selected nodes to the given ax object
    # with a given title
    def plot_V(self, ax: mpl.axes.Axes, title: str, pattern: list) -> None:
        V = self.V[:, :, 0].detach().cpu()
        for i in self.targetOscs:
            if pattern[i] == 1:
                clr = 'r'
                lbl = "+"
            else:
                clr = 'b'
                lbl = '-'
            ax.plot(self.tSpace[-self.plotCO:].cpu(),
                    V[-self.plotCO:, i*self.invNum+self.targetNodes[i]],
                    label=lbl, color=clr)

        ax.set_title(title, fontsize='large', fontweight='bold')
        ax.set_xlabel("t (s)", fontsize='large', fontweight='bold')
        ax.set_ylabel("Voltage at outputs (V)", fontsize='large',
                      fontweight='bold')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.8])

        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in
                  enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

        return

    # It plots the target oscillatory system's target nodes
    # and saving to figName if save is set to True
    def plot_results(self, figID: int = 0, figName: str = "Res",
                     subplotRnum: int = 1, subplotCnum: int = 1,
                     subplotPlace: int = 1, title: str = "",
                     pattern: list = None) -> None:
        plt.figure(figID)

        ax = plt.subplot(subplotRnum, subplotCnum, subplotPlace)

        if pattern is None:
            pattern = self.pattern

        self.plot_V(ax, title, pattern)

        if not self.cluster:
            plt.show(block=False)

        if self.save:
            self.fig.savefig(self.saveFolder + figName + ".png")

            self.fig.close()

        return

    # This builds up the graph from the coupling matrix
    def build_graph(self):
        self.G = nx.Graph()
        for i in range(self.oscNum):
            self.G.add_node(i)

        for i in range(1, self.oscNum):
            for j in range(i+1, self.oscNum):
                if self.Couplings[i, j] != 0:
                    self.G.add_edge(i, j, weight=self.Couplings[i, j])

        return

    # This is used to plot the coupled oscillator system as a graph
    # If save is set to True, it also saves the plot to figName
    def plot_graph(self, figName: str = "Graph") -> None:
        self.build_graph()

        plt.figure(self.figGraph)
        plt.title('Coupled oscillator system with the coupling weights')
        nx.draw_networkx(self.G, pos=nx.circular_layout(self.G), node_size=500)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos=nx.circular_layout(self.G),
                                     edge_labels=labels)

        plt.axis("off")

        if self.save:
            plt.savefig(self.saveFolder + figName + ".png")

            plt.close()

        return


class COCRegular(COCAbstract):

    # Constructor for the COCRegular class
    def __init__(self, dateString: str) -> None:
        super().__init__(dateString)

        return

    # This overrides the build_from_file method to also create Cprime
    def build_from_file(self, fileName: str = "default.in") -> None:
        super().build_from_file(fileName)

        # Create the Cprime and Bprime matrices for the calculations
        self.create_Cprime()
        self.create_Bprime()
        
        # Transform inputs to coincide with implementation
        # self.inputs = self.inputs[:, np.newaxis]

        # Create extra fig identifier
        self.figConfigs = 3

        return

    # Construction of Bprime for inputs
    def create_Bprime(self) -> None:
        # B: humanly readable B Matrix
        # Bprime: constructed B matrix for the equations
        self.Bprime = torch.zeros(self.invNum*self.oscNum, self.inputNum,
                                  dtype=torch.float64)
        for i in range(self.oscNum):
            for j in range(self.inputNum):
                if self.B[i, j] != 0:
                    self.Bprime[i*self.invNum, j] = self.B[i, j]

        write_tensor("Bprime.txt", self.Bprime, self.oscNum, self.invNum)
        
        self.Bprime = self.Bprime.to(self.device)
        
        return

    # Construction of Cprime for Couplings
    def create_Cprime_from_C(self) -> None:
        # Couplings: C (in notes), coupling matrix
        # Cprime: constructed couplings matrix
        self.Cprime = torch.zeros(self.invNum*self.oscNum,
                                  self.invNum*self.oscNum,
                                  dtype=torch.float64)
        for i in range(self.oscNum):
            for j in range(i+1, self.oscNum):
                if self.Couplings[i, j] == 1:
                    self.Cprime[i*self.invNum + 2, i*self.invNum + 2] -= \
                        self.Couplings[i, j]
                    self.Cprime[i*self.invNum + 2, j*self.invNum + 2] = \
                        self.Couplings[i, j]
                    self.Cprime[j*self.invNum + 2, i*self.invNum + 2] = \
                        self.Couplings[i, j]
                elif self.Couplings[i, j] == -1:
                    self.Cprime[i*self.invNum + 5, i*self.invNum + 5] -= \
                        abs(self.Couplings[i, j])
                    self.Cprime[j*self.invNum + 2, i*self.invNum + 5] = \
                        abs(self.Couplings[i, j])
                    self.Cprime[i*self.invNum + 5, j*self.invNum + 2] = \
                        abs(self.Couplings[i, j])

                if self.Couplings[i, j] != 0:
                    self.Cprime[j*self.invNum + 2, j*self.invNum + 2] -= \
                        abs(self.Couplings[i, j])

        write_tensor("Cprime.txt", self.Cprime, self.oscNum, self.invNum)
        
        self.Cprime = self.Cprime.to(self.device)

        return

    # This is a simple ode integration
    @timeit("The elapsed time of the integration is %.3fs")
    def run_simulation(self) -> None:
        self.progressBar.start()

        self.V = odeint(self.ode_fn, self.V_init, self.tSpace).to(self.device)

        self.eqs_to_txt()

        if hasattr(self.inputs.T):
            pattern = str(list(self.inputs.T))
            patternString = "for pattern: %s" % pattern
        else:
            patternString = ""
        title = "Voltages at selected nodes of select oscillators" + patternString
        self.plot_results(self.figRes, subplotRnum=1, subplotCnum=1,
                          subplotPlace=1, title=title, pattern=self.inputs)

        return

    # This calculates every energy configuration for a given model
    def calc_all_configs(self) -> None:
        self.Hs = {}
        for i in range(2**self.oscNum):
            actStr = np.binary_repr(i, width=self.oscNum)
            act = 2*np.array(list(map(int, actStr))) - 1

            self.Hs[actStr] = -np.matmul(act, np.matmul(self.Couplings, act))

        return

    # This plots all the energy configuration of a given system
    def plot_configs(self, figName: str = "Energy") -> None:
        self.calc_all_configs()

        plt.figure()

        plt.bar(range(len(self.Hs)), list(self.Hs.values()),
                align='center')
        plt.xticks(range(len(self.Hs)), list(self.Hs.keys()), rotation=90)

        plt.title("Hamiltonian of all the possible configuration")
        plt.xlabel("Configuration")
        plt.ylabel("H")

        if self.save:
            plt.savefig(self.saveFolder + figName + ".png")

            plt.close()

        return


class COCLearnableAbstract(COCAbstract, ABC):

    # Inner class for the actual model in the training
    class Model(nn.Module):
        def __init__(self, odefn, oscNum) -> None:
            super().__init__()

            self.Couplings = nn.Parameter(torch.rand((oscNum, oscNum)))
            self.B = torch.full((oscNum,), 1.0)

            self.ode_fn = odefn

            return

        # This projects the parameters back to the feasible set
        def project_params(self) -> None:
            torch.clamp(self.Couplings.data, min=0.0, out=self.Couplings.data)
            torch.clamp(self.B.data, min=0.0, out=self.B.data)

            return
    
        # This prunes the parameters according to a given methodology
        def prune_params(self) -> None:
            self.Couplings.data[self.Couplings <= 0.2] = 0.0
            self.B.data[self.B <= 0.2] = 0.0

            return

        # This is called for a step in the learning process propagation
        def forward(self, t: float, V: torch.tensor) -> torch.tensor:
            self.progressBar.update(t)

            dV = self.ode_fn(t, V)

            return dV

    # Constructor for the COCLearnable class
    def __init__(self, dateString: str, batchNum: int, epochs: int, type: str) -> None:
        super().__init__(dateString)
        
        self.batchNum = batchNum
        self.epochs = epochs
        self.type = type

        return

    # Overriding the function to expand with the learnable parameters
    def build_from_file(self, fileName: str = "default.in") -> None:
        super().build_from_file(fileName)

        # Creating the model for the training (the actual model is the odefn!)
        self.model = self.Model(self.ode_fn, self.oscNum).to(self.device)

        # Send the time array to the appropriate device
        self.tSpace = self.tSpace

        # Init optimizer for the learning process
        self.optimizer = optim.Adam(self.model.parameters(), lr=1E-2)

        # Create Cprime and Bprime initially
        self.create_Cprime()
        self.create_Bprime()

        # Set the default value for epochs
        if not hasattr(self, "epochs"):
            self.epochs = 200 

        # Parameters for the learning process
        self.Losses = torch.zeros(self.epochs)
        self.paramsInTime = np.zeros(
            (self.epochs,) + self.model.Couplings.shape
        )
        self.cps = np.linspace(0, self.epochs-1, 10, dtype=int)

        # Loss calculation cutoff
        # This is again arbitrary and means that the loss value is calculated
        # from the result where we trim the first 40% of the data (due to transients)
        self.lossCO = int(0.6*self.tNsamples)

        # Create extra fig identifiers
        self.figLearning = 4
        self.figLoss = 5

        return

    # Construction of Cprime for Couplings
    def create_Cprime_full(self) -> None:
        # Couplings: C (in notes), coupling matrix
        # Cprime: constructed couplings matrix
        # from Coupings
        self.Cprime = torch.zeros(self.oscNum*self.invNum,
                                  self.oscNum*self.invNum,
                                  dtype=torch.float64, device=self.device)
        for i in range(self.oscNum-1):
            for j in range(i+1, self.oscNum):
                # Positive couplings
                self.Cprime[i*self.invNum + 2, i*self.invNum + 2] -= \
                    self.model.Couplings[i, j]
                self.Cprime[i*self.invNum + 2, j*self.invNum + 2] = \
                    self.model.Couplings[i, j]
                self.Cprime[j*self.invNum + 2, i*self.invNum + 2] = \
                    self.model.Couplings[i, j]
                self.Cprime[j*self.invNum + 2, j*self.invNum + 2] -= \
                    self.model.Couplings[i, j]

                # Negative couplings
                self.Cprime[i*self.invNum + 5, i*self.invNum + 5] -= \
                    self.model.Couplings[j, i]
                self.Cprime[i*self.invNum + 5, j*self.invNum + 2] = \
                    self.model.Couplings[j, i]
                self.Cprime[j*self.invNum + 2, i*self.invNum + 5] = \
                    self.model.Couplings[j, i]
                self.Cprime[j*self.invNum + 2, j*self.invNum + 2] -= \
                    self.model.Couplings[j, i]

        write_tensor("Cprime.txt", self.Cprime, self.oscNum, self.invNum)

        return
    
    # This create a Cprime matrix corresponding to neareast neighbour connection
    # Note that the oscillators's voltages are still in a column vector, but 
    # the representation is row-wise 
    # Maybe it would be better to adapt the equations for different layouts??
    def create_Cprime_nearest_neighbour(self) -> None:
        return

    # Overriding create_Bprime to correspond with learnable class
    def create_Bprime(self):
        # Bprime: constructed B matrix for the equations
        self.Bprime = torch.zeros(self.invNum*self.oscNum, self.inputNum,
                                  dtype=torch.float64, device=self.device)
        for i in range(self.inputNum):
            self.Bprime[i*self.invNum + 2, i] = self.model.B[i]

        write_tensor("Bprime.txt", self.Bprime, self.oscNum, self.invNum)

        return

    # Run single simulation
    # Note that it also creates a new V_init to make it a stochastic gradient
    # algorithm instead of a regular gradient one
    @timeit("The elapsed time of the integration %.3fs")
    def single_run(self) -> None:
        self.V = odeint(
            self.model,
            self.V_init,
            self.tSpace,
            method=self.method
        ).to(self.device)

        return

    # This is an abstract method for calculating the loss
    @abstractmethod
    def calc_loss(self, V, *others):
        pass

    @timeit("The elapsed time of the training is %.3fs")
    # Calculate one learning step and also measures a step for one learning step
    def learn(self, loss: torch.tensor) -> torch.tensor:
        loss.backward()
        self.optimizer.step()

        self.model.project_params()

        self.create_Cprime()
        self.create_Bprime()

        return loss

    # This post-processes the parameters to exclude NaN due to zero division
    # and also convert the parameters to resistance values
    def post_process(self) -> None:
        with np.errstate(divide="ignore"):
            self.paramsInTime = self.Rc / self.paramsInTime

        # Removes reasonably high resistances
        # This is a post process pruning
        self.paramsInTime[self.paramsInTime >= 50000] = 0.0

        return

    # This plots the learning curve (LC) and the parameters with respect
    # to the number of simulations that have been done
    # If save is set to True, it also save the figure to figName
    def plot_LC_w_params(self, figName: str = "LossWParam") -> None:
        fig = plt.figure(self.figLoss)
        ax = fig.add_subplot(1, 1, 1)

        color = "tab:blue"
        ax.plot(self.Losses, 'b')
        ax.set_xlabel("# of simulations")
        ax.set_ylabel("Loss")
        ax.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()

        if self.save:
            plt.savefig(self.saveFolder + figName + ".png")

            plt.close()

        return

    # This helps the visualisation of the training by plotting the means
    def plot_means(self, ax: mpl.axes.Axes) -> None:
        ax.plot(self.tSpace[-self.plotCO:].cpu(),
                self.meanMns[-self.plotCO:, 0].detach().cpu())
        ax.plot(self.tSpace[-self.plotCO:].cpu(),
                self.meanPls[-self.plotCO:, 0].detach().cpu())

        x = torch.sum(torch.square(self.meanMns - self.meanPls))
        ax.set_title("Group means with %f difference" % x,
                     fontsize='large', fontweight='bold')
        ax.set_xlabel("time (s)", fontsize='large', fontweight='bold')
        ax.set_ylabel("Mean voltages (V)", fontsize='large', fontweight='bold')
        ax.legend(["Minus group", "Plus group"])

        return

    # This helps visualising the learning process by plotting the actual output
    def visualise_training(self, i: int, y: int = None) -> None:
        fig = plt.figure(self.figLearning)

        plt.clf()

        ax = fig.add_subplot(2, 1, 2)
        self.plot_means(ax)

        if y is None:
            pattern = self.pattern
            patternString = str(list(pattern))
            title = ("Voltages at %i." % i) + \
                ("iteration with %s" % (patternString)) + " pattern"
        else:
            pattern = getattr(self, "pattern" + str(y))
            patternString = str(list(pattern))
            title = ("Voltages at %i." % i) + \
                ("iteration with %s" % (patternString)) + " as pattern%i" % y

        self.plot_results(self.figLearning, subplotRnum=2, subplotCnum=1,
                          subplotPlace=1, title=title, pattern=pattern)

        if i in self.cps:
            plt.savefig(self.saveFolder + "Results_%i_it" % i + ".png")

        plt.tight_layout()

        plt.draw()
        plt.pause(0.001)

        return

    # This prints the resistance values to a file
    def couplings_to_file(self, fileName: str) -> None:
        parameterFile = self.saveFolder + fileName + ".pmts"

        torch.save(self.paramsInTime, parameterFile)

        return

    # This builds up the graph from the coupling matrix
    # This is an override of the original build_graph method for this class
    def build_graph(self) -> None:
        self.G = nx.Graph()
        self.G.add_nodes_from(range(1, self.oscNum+1))

        for i in range(self.oscNum):
            for j in range(i+1, self.oscNum):
                if self.paramsInTime[-1, i, j] != 0.0:
                    plusRstr = "%.1fkΩ" % (self.paramsInTime[-1, i, j] / 1000)
                else:
                    plusRstr = ""
                if self.paramsInTime[-1, j, i] != 0.0:
                    negRstr = "%.1fkΩ" % (self.paramsInTime[-1, j, i] / 1000)
                else:
                    negRstr = ""
                weightStr = "%s\n%s" % (plusRstr, negRstr)

                if weightStr != '\n':
                    self.G.add_edge(i+1, j+1, weight=weightStr)

        return

    # This is used to plot the coupled oscillator system as a graph
    # Also, this is overrides the abstract classes plot_graph method
    def plot_graph(self, figName: str = "Graph") -> None:
        self.build_graph()

        plt.figure(self.figGraph)
        plt.title('Coupled oscillator system with the coupling weights')
        nx.draw_networkx(self.G, pos=nx.circular_layout(self.G), node_size=500)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos=nx.circular_layout(self.G),
                                     edge_labels=labels)

        plt.axis("off")

        if self.save:
            plt.savefig(self.saveFolder + figName + ".png")

            plt.close()

        return


class COCPatternRecognition(COCLearnableAbstract):

    # Constructor for the COCPatternRecognition class
    def __init__(self, dateString: str, batchNum: int, epochs: int, type: str, classNum: int) -> None:
        super().__init__(dateString, batchNum, epochs, type)
        
        self.classNum = classNum

        return

    # Overriding the function to expand with the learnable parameters
    def build_from_file(self, fileName: str = "default.in") -> None:
        super().build_from_file(fileName)

        # Setting the patterns based on its mode
        self.patterns = set()
        while len(self.patterns) < self.classNum:
            patternCandidate = torch.randint(2, (self.oscNum,))
            if patternCandidate not in self.patterns:
                self.patterns.add(tuple(patternCandidate.tolist()))

        self.patterns = list(self.patterns)

        for pattern in self.patterns:
            print(pattern)

        # Set figure numbers
        self.figTest = 6

        return

    # Create input from pattern
    def create_inputs_by_pattern(self, y: list) -> None:
        self.inputs = torch.zeros(self.inputNum, self.batchNum,
                                  dtype=torch.int64)

        for i, cls in enumerate(y):
            pattern = torch.DoubleTensor(self.patterns[cls])
            self.inputs[:, i] = pattern
            
        self.inputs = self.inputs.to(self.device)
        
        return

    # This calculate the difference in the same pattern group and also
    # their mean signal value for the intergroup difference calculation
    def calc_intragroup_diff_and_mean(self, indices, V) \
            -> Tuple[torch.tensor, torch.tensor]:
        groupSize = len(indices)

        loss = torch.zeros(1, V.shape[2],
                           dtype=torch.float64, device=self.device)
        mean = torch.zeros(self.lossCO, V.shape[2],
                           dtype=torch.float64, device=self.device)

        if groupSize == 0:
            return torch.zeros(1), mean

        if groupSize == 1:
            ind = indices[0]
            return torch.zeros(1), \
                V[-self.lossCO:, ind*self.invNum + self.targetNodes[ind], :]

        for i in indices:
            osc_i = V[-self.lossCO:, i*self.invNum + self.targetNodes[i], :]
            mean += osc_i
            for j in indices:
                if j <= i:
                    continue
                osc_j = \
                    V[-self.lossCO:, j * self.invNum + self.targetNodes[j], :]
                diff = osc_i - osc_j

                loss += torch.sum(torch.square(diff), 0) / self.lossCO
        '''
        # v3 intragroup loss
        for i in indices:
            osc_i = self.V[-co:, i*self.invNum + self.targetNodes[i]]
            mean += osc_i

        mean *= 1 / groupSize

        for i in indices:
            osc_i = self.V[-co:, i*self.invNum + self.targetNodes[i]]
            diff = mean - osc_i
            loss += torch.sum(torch.square(diff)) / co

        loss *= 1 / groupSize
        '''

        loss = loss / (groupSize * (groupSize-1) / 2)
        loss = torch.sum(loss / self.batchNum)

        mean = mean / groupSize

        return loss, mean

    # This calculate the complex loss as three different parts which are:
    # a: intragroup difference in the + group
    # b: intragroup differnece in the - group
    # c: intergroup difference between + and - with the applicaiton of 1/x
    def calc_loss(self, V, *otherArgs) -> torch.tensor:
        # 3 parts: intragroup diff +, intragroup diff -, intergroup diff with f
        # 3 parameters for scaling
        a = torch.tensor([3.0], device=self.device)
        b = torch.tensor([3.0], device=self.device)
        c = torch.tensor([40.0], device=self.device)

        pattern = otherArgs[0]

        # Intragroup indices
        indMns = [i for i, e in enumerate(pattern) if e == 0]
        indPls = [i for i, e in enumerate(pattern) if e == 1]

        # Intragroup differences and means
        lossMns, self.meanMns = self.calc_intragroup_diff_and_mean(indMns, V)
        lossPls, self.meanPls = self.calc_intragroup_diff_and_mean(indPls, V)

        # Intergroup difference between + and -
        x = torch.sum(torch.square(self.meanMns - self.meanPls), 0)
        lossIntergroupDiff = torch.sum(1/x) / self.batchNum
        
        print("Minus group loss: %f" % (a*lossMns).item())
        print("Plus group loss: %f" % (b*lossPls).item())
        print("Inter Group loss: %f" % (c*lossIntergroupDiff).item())

        # Calculate the linear combination of losses with a, b, c as coeffs
        loss = a*lossMns + b*lossPls + c*lossIntergroupDiff

        return loss

    # This runs the learning process for the system
    def run_simulation(self) -> None:
        for i in range(self.epochs):
            self.optimizer.zero_grad()

            self.paramsInTime[i, :, :] = \
                self.model.Couplings[:, :].detach().cpu()

            # 0 or 1 as input classes if classes = 2
            y = torch.randint(self.classNum, (self.batchNum,)).tolist()
            
            self.create_random_V_init()

            self.create_inputs_by_pattern(y)

            self.progressBar.set_string("The %i. iteration: " % (i+1))
            self.progressBar.start()
            self.model.progressBar = self.progressBar

            self.single_run()

            self.progressBar.reset()

            for cls in range(self.classNum):
                actIndices = [i for i, e in enumerate(y) if e == cls]
                actPattern = self.patterns[cls]

                if len(actIndices) != 0:
                    print("The losses for the group %i:" % cls)
                    if cls == 0:
                        loss = self.calc_loss(self.V[:, :, actIndices],
                                              actPattern)
                    else:
                        loss += self.calc_loss(self.V[:, :, actIndices],
                                               actPattern)

            self.Losses[i] = loss.item()
            loss = self.learn(loss)

            print("The loss is %.3f." % self.Losses[i])

            if i in self.cps:
                torch.save({
                        "epoch": self.epochs,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss
                    }, (self.saveFolder + "model_cps/model_at_%i" % i)
                )

        self.post_process()

        self.test_network()

        self.model.prune_params()
        self.create_Cprime()
        self.create_Bprime()

        self.test_network(True)

        return

    # Runs the integration for the trained network to test its performance
    def test_network(self, afterPruning: bool = False, figName: str = "Test_"):
        plt.figure(self.figTest)

        saveBatch = self.batchNum
        self.batchNum = 1

        plotOffset = self.classNum if afterPruning else 0
        print(plotOffset)

        for i, pattern in enumerate(self.patterns):
            ax = plt.subplot(2, self.classNum, i+1+plotOffset)

            self.create_random_V_init()
            self.create_inputs_by_pattern([i])

            self.progressBar.set_string("The test for class %i: " % i)
            self.progressBar.start()
            self.model.progressBar = self.progressBar

            self.single_run()

            self.progressBar.reset()

            titleAddition = "after pruning" if afterPruning else ""

            title = "Result for class %i with pattern: %s %s" \
                    % (i, str(list(pattern)), titleAddition)
            print(pattern)
            self.plot_V(ax, title, pattern)

            self.save_results("MLResult_" + str(i) + "_postpruning_")

        self.batchNum = saveBatch

        if afterPruning and self.save:
            plt.savefig(self.saveFolder + figName + ".png")

            plt.close()

        return

    # Override the visualise_training method for the pattern recognition module
    # TODO: implement a creative way to visualise training for multiple patterns
    def visualise_training(self) -> None:
        return

# TODO: implement the whole class
class COCPatternGeneration(COCLearnableAbstract):

    # Constructor for the COCPatternGeneration class
    def __init__(self, dateString: str) -> None:
        super().__init__(dateString)

        return

    # Overriding the function to expand with the learnable parameters
    def build_from_file(self, fileName: str = "default.in") -> None:
        super().build_from_file(fileName)

        return

    # This sets the target patter for the simulation
    def set_pattern(self, filename: str = "default.pat") -> None:

        return

    # Overriding the parent's function
    def calc_loss(self, V):
        return

    # Overriding run_simulation
    def run_simulation(self) -> None:
        return
