import cytnx
import numpy as np
import itertools
import time
from exact_free_energy import free_energy_per_site

def initial_TN(temp: float):
    M = np.array([[+np.sqrt(np.cosh(+1/temp)), +np.sqrt(np.sinh(+1/temp))],
                [+np.sqrt(np.cosh(+1/temp)), -np.sqrt(np.sinh(+1/temp))]])
    bd = cytnx.Bond(2)
    T = cytnx.UniTensor([bd,bd,bd,bd], rowrank=4).set_name("T").relabel(["u","r","d","l"])
    for u,r,d,l in itertools.product([0, 1], repeat=4):
        T.at(["u","r","d","l"], [u,r,d,l]).value = M[0,u]*M[0,r]*M[0,d]*M[0,l] + M[1,u]*M[1,r]*M[1,d]*M[1,l]
    # T.print_diagram()
    trT = T.Trace("u","d").Trace("r","l").item()
    T /= trT
    log_factor = np.log(trT)
    n_spin = 1

    return (T, log_factor, n_spin)

class TRG:
    def __init__(self, temp: float, chi: int) -> None:
        self.temp = temp
        self.chi = chi
        a, log_factor, n_spin = initial_TN(self.temp)
        self.A = a
        self.log_factors = [log_factor]
        self.n_spins = [n_spin]
        self.step = 0

    def print_elapsed_time(self, elapsed_time: float) -> None:
        print(f"# Elapsed time: {elapsed_time:.6f} sec")

    def update(self) -> None:
        T = self.A.set_rowrank(2)

        T = T.permute(['u','r','d','l'])
        # T.print_diagram()
        # S , U , Vdag = cytnx.linalg.Svd(T)
        S , U , Vdag = cytnx.linalg.Svd_truncate(T, keepdim=self.chi)
        # print("S=",S.get_block().numpy())    
        # U[:,:,2] *= -1
        # U[:,:,3] *= -1
        # print("U=\n",U.get_block().numpy())    
        # Vdag[2,:,:] *= -1
        # Vdag[3,:,:] *= -1 
        # print("Vdag=\n",Vdag.get_block().numpy())
        S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
        # print("S_sqrt=\n",S_sqrt.get_block().numpy())
        C3 = cytnx.Contract(U, S_sqrt).set_name("C3")
        C1 = cytnx.Contract(S_sqrt, Vdag).set_name("C1")
        # C3.print_diagram()
        # C1.print_diagram()

        T = T.permute(['u','l','r','d'])
        # T.print_diagram()
        # S , U , Vdag = cytnx.linalg.Svd(T)
        S , U , Vdag = cytnx.linalg.Svd_truncate(T, keepdim=self.chi)
        # print("S=\n",S.get_block().numpy())
        # print("U=\n",U.get_block().numpy())
        # print("Vdag=\n",Vdag.get_block().numpy())
        S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
        # print("S_sqrt=\n",S_sqrt.get_block().numpy())
        C2 = cytnx.Contract(U, S_sqrt).set_name("C2")
        C0 = cytnx.Contract(S_sqrt, Vdag).set_name("C0")
        # C2.print_diagram()
        # C0.print_diagram()

        from extension import conc
        tensors = [("C0", ["aux","r","d"]), ("C1", ["aux","d","l"]), ("C2", ["u","l","aux"]), ("C3", ["u","r","aux"])]
        contractions = [("C0", "r", "C1", "l"),("C1","d","C2","u"),("C2","l","C3","r"),("C3","u","C0","d")]
        net, net_string = conc(tensors, contractions)
        # print(net)
        net.PutUniTensor("C0", C0, ["_aux_L","r","d"])
        net.PutUniTensor("C1", C1, ["_aux_L","d","l"])
        net.PutUniTensor("C2", C2, ["u","l","_aux_R"])
        net.PutUniTensor("C3", C3, ["u","r","_aux_R"])
        # print(net)
        TT = net.Launch().set_name("TT").relabel(["u","r","d","l"])
        # TT.print_diagram()
        trTT = TT.Trace("u","d").Trace("r","l").item()
        # print("trTT=",trTT)

        # TT = net.Launch().set_name("TT")
        # TT.print_diagram()
        # trTT = TT.Trace("C0_aux","C2_aux").Trace("C1_aux","C3_aux").item()
        # print("trTT=",trTT)

        # normalize
        factor = trTT
        # print("factor=",factor)
        self.A = TT/factor
        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])
        self.step += 1

    def run(self, step: int) -> None:
        exact = free_energy_per_site(self.temp)

        time_start = time.perf_counter()
        f = -np.sum(np.array(self.log_factors) / np.array(self.n_spins)) * self.temp
        f_err = (f - exact) / abs(exact)
        print(f"{self.step:04d}", f"{self.n_spins[-1]:6d}", f"{f:.12e}", f"{f_err:.12e}")
        for i in range(step):
            self.update()
            # self.print_results()
            f = -np.sum(np.array(self.log_factors) / np.array(self.n_spins)) * self.temp
            f_err = (f - exact) / abs(exact)
            print(f"{self.step:04d}", f"{self.n_spins[-1]:6d}", f"{f:.12e}", f"{f_err:.12e}")

        time_end = time.perf_counter()
        self.print_elapsed_time(time_end - time_start)


########################################################
#  Levin-Nave TRG for 2D Ising model on square lattice
########################################################
Tc = 2/np.log(1+np.sqrt(2))
temp = Tc
step = 20

for chi in [4,8,16,40]:
    print("temp =", temp, "step =", step, "chi =", chi)
    TRG(temp, chi).run(step)

