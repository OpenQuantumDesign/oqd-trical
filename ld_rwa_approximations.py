from quantumion.compilerv2 import Post, RewriteRule, FixedPoint
from analog_Hamiltonian_AST import OperatorScalarMul, OperatorMul, OperatorKron, WaveCoefficient, ApproxDisplacementMatrix, Displacement

import numpy as np

class ReorderScalarMul(RewriteRule):
    def map_OperatorMul(self, model):
        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, OperatorScalarMul) and isinstance(op2, OperatorKron):

            
            if isinstance(op1.op, OperatorKron) and isinstance(op1.coeff, WaveCoefficient):
                coeff = op1.coeff
                int_term = op1.op
                mot_term = op2

                return OperatorMul(op1 = int_term, op2 = OperatorScalarMul(coeff = coeff, op = mot_term))

        else:
            return model


def approximate(tree, n_cutoff, timescale, ld_cond_th = 1e-2, rwa_cutoff = 'inf'):
    
    reorder = Post(ReorderScalarMul())

    reordered_tree = reorder(tree)

    approximator = Post(RWA_and_LD_Approximations(n_cutoff=n_cutoff, rwa_cutoff=rwa_cutoff, timescale=timescale, ld_cond_th=ld_cond_th))

    approx_tree = approximator(reordered_tree)
    

    return approx_tree


class approx_ekron(RewriteRule):

    def __init__(self, Delta, n_cutoff, rwa_cutoff, ld_cond_th = 1e-2):
        super().__init__()
        self.Delta = Delta
        self.n_cutoff = n_cutoff
        self.rwa_cutoff = rwa_cutoff
        self.ld_cond_th = ld_cond_th

    def map_Displacement(self, model):
        
        alpha = model.alpha
        dims = model.dims
        eta = alpha.amplitude
        nu = alpha.frequency

        if eta**2 * (2*self.n_cutoff + 1)< self.ld_cond_th:
            # ^ First order condition
            ld_order =  1
            print("Expanding to 1st order in the Lamb-Dicke approximation")
    
        elif 2*eta**4 * (self.n_cutoff**2 + self.n_cutoff + 1) < self.ld_cond_th:
            # ^ Second order condition
            ld_order = 2
            print("Expanding to 2nd order in the Lamb-Dicke approximation")
        else:
            ld_order = 3
            print("Expanding to 3rd order in the Lamb-Dicke approximation")

        return ApproxDisplacementMatrix(ld_order = ld_order, alpha = alpha, rwa_cutoff = self.rwa_cutoff, 
                                        Delta = self.Delta, nu = nu, dims = dims)


class RWA_and_LD_Approximations(RewriteRule):
    
    # Performs both RWA and LD approximation using helper function in utilities
    
    def __init__(self, n_cutoff, rwa_cutoff, timescale, ld_cond_th = 1e-2):
        super().__init__()
        self.n_cutoff = n_cutoff
        self.rwa_cutoff = rwa_cutoff
        self.timescale = timescale
        self.ld_cond_th = ld_cond_th

    def map_OperatorScalarMul(self, model):
        
        coeff = model.coeff
        op = model.op
        
        if isinstance(op, OperatorKron) and isinstance(coeff, WaveCoefficient):
            
            # At this point we know that op corresponds to the mot_term part of the Hamiltonian
            # We'd like to now pass this into a rewrite rule that replaces D(\alpha) with the approximation

            Delta = -coeff.frequency

            if isinstance(self.rwa_cutoff, float):
                rwa_cutoff = 2*np.pi*self.rwa_cutoff * self.timescale
            else:
                rwa_cutoff = self.rwa_cutoff

            ekron_approximator = Post(approx_ekron(Delta = Delta, n_cutoff=self.n_cutoff, rwa_cutoff=rwa_cutoff, ld_cond_th=self.ld_cond_th))

            approx_mot = ekron_approximator(op)

            return OperatorScalarMul(coeff = coeff, op = approx_mot)
        else:
            return model

            
            kron_op1 = op.op1
            kron_op2 = op.op2
            print("halo")
            print("op1:", kron_op1)

            print("op2:", kron_op2)

            if isinstance(kron_op1, Displacement) or isinstance(kron_op2, Displacement):
                print("balo")


                
                Delta = -coeff.frequency # already scaled and angular; minus: there's a minus sign in the non-hc exp in Hamiltonian

                if isinstance(kron_op1, Displacement):
                    d_op = kron_op1
                    eta = d_op.alpha.amplitude
                    nu = d_op.alpha.frequency
                    


                    if eta**2 * (2*self.n_cutoff + 1)< self.ld_cond_th:
                        # ^ First order condition
                        ld_order =  1
                
                    elif eta**4 * (4*self.n_cutoff**2 + 4*self.n_cutoff + 3) < self.ld_cond_th:
                        # ^ Second order condition
                        ld_order = 2
                    else:
                        ld_order = 3

                    print("A")

                    return OperatorKron(op1 = OperatorScalarMul(op = ApproxDisplacementMatrix(ld_order = ld_order, alpha = d_op.alpha, rwa_cutoff = rwa_cutoff, 
                                                    Delta = Delta, nu = nu, dims = d_op.dims),
                                            coeff = coeff), op2 = kron_op2)
                
                elif isinstance(kron_op2, Displacement):
                    d_op = kron_op2
                    eta = d_op.alpha.amplitude
                    nu = d_op.alpha.frequency

                    if eta**2 * (2*self.n_cutoff + 1)< self.ld_cond_th:
                        # ^ First order condition
                        ld_order =  1
                
                    elif eta**4 * (4*self.n_cutoff**2 + 4*self.n_cutoff + 3) < self.ld_cond_th:
                        # ^ Second order condition
                        ld_order = 2
                    else:
                        ld_order = 3

                    print("B")

                    return OperatorKron(op1 = kron_op1, op2 = OperatorScalarMul(op = ApproxDisplacementMatrix(ld_order = ld_order, alpha = d_op.alpha, rwa_cutoff = rwa_cutoff, 
                                                    Delta= Delta, nu = nu, dims = d_op.dims),
                                            coeff = coeff))