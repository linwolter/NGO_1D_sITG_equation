import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum

#Function taken from work by Joost Prins
def torch_repeat(x, repeats, axis=None):
    """
    PyTorch equivalent of np.repeat.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    repeats : int or 1D list/Tensor of ints
        Number of repetitions.
    axis : int or None
        Dimension along which to repeat. If None, the tensor is flattened.

    Returns
    -------
    torch.Tensor
    """

    # Match NumPy: if axis is None, flatten first
    if axis is None:
        x = x.flatten()
        return torch.repeat_interleave(x, repeats)

    # Normalize negative axes
    if axis < 0:
        axis += x.dim()

    # Move target axis to the end
    x = x.transpose(axis, -1)

    # Apply repeat_interleave on last dimension
    out = torch.repeat_interleave(x, repeats, dim=-1)

    # Move axis back
    out = out.transpose(axis, -1)

    return out


# Class taken from work by Joost Prins
class BSplineBasis:
    """
    B-spline basis functions with vectorized evaluation and differentiation using PyTorch.

    Attributes:
        N (int): Number of basis functions.
        p (int): Degree of the B-splines.
        C (int): Continuity at internal knots (C < p).
        t (torch.Tensor): Knot vector.
        L (float): Scaling to parent domain [0,L].
        Dx (float): Translation to parent domain if needed.
        dtype (torch.dtype): Torch data type for computations.
        device (torch.device): Torch device for computations.
    """
    def __init__(self, N, p, C, dtype, device):
        """""
        Initialize the B-spline basis.

        Args:
            N: Number of basis functions.
            p: Degree of the B-splines.
            C: Continuity at internal knots (C < p).
            dtype: Torch data type for computations.
            device: Torch device for computations.
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.N = N
        self.L = 1.0  # Scaling to parent domain [0,L]
        self.Dx = 0 # Translation to parent domain if needed
        self.p = p
        self.C = C
        # Knot vector
         # Repeated knots at start
        self.t = torch.zeros(self.p+1, dtype=self.dtype, device=self.device)
        # Equally spaced internal knots with multiplicity (p - C)
        self.t = torch.cat((
            self.t,
            torch_repeat(
                torch.linspace(
                    0, 1,
                    int((self.N - self.p - 1)/(self.p - self.C)) + 2,
                    dtype=self.dtype, device=self.device
                )[1:-1],
                self.p - self.C
            )
        ))
        # Repeated knots at end
        self.t = torch.cat((self.t, torch.ones(self.p+1, dtype=self.dtype, device=self.device)))
        # Set tolerances based on dtype
        if self.dtype == torch.float32:
            self.rtol, self.atol = 1e-5, 1e-7
        if self.dtype == torch.float64:
            self.rtol, self.atol = 1e-13, 1e-15

    def to(self, dtype, device):
        self.__init__(self.N, self.p, self.C, dtype, device)

    def _basis_dp(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """
        Vectorized Cox–de Boor evaluation for all basis functions in parallel.

        Args:
            x: 1D tensor of evaluation points, shape (Q,).
            return_prev: If True, also returns I_{*,p-1}(x) for derivative.

        Returns:
            I_p: (N, Q) values of I_{i,p}(x).
            I_pm1: (N, Q) values of I_{i,p-1}(x) if return_prev=True, else None.
        """
        # ------------------------------------------------------
        # Degree 0 basis: indicator functions:
        #
        #   I_{i,0}(x) = 1  if  t_i ≤ x < t_{i+1}
        #                0  otherwise
        #
        # This produces an (N,Q) boolean tensor.
        # ------------------------------------------------------
        t_i = self.t[:self.N].unsqueeze(1)       # (N,1)
        t_ip1 = self.t[1:self.N+1].unsqueeze(1)  # (N,1)
        # Use half-open intervals [t_i, t_{i+1})
        I = (t_i <= x) & (x < t_ip1)  # (N,Q) boolean
        # Right-clamping allows the last basis to include x == t_{N+p+1}
        at_right_endpoint = torch.isclose(x, t_ip1[-1], rtol=self.rtol, atol=self.atol)
        # Left-clamping for first basis function
        I[-1] = I[-1] | at_right_endpoint
        
        # If degree is zero, stop here.
        if p == 0:
            return I

        # ------------------------------------------------------
        # Cox–de Boor recurrence:
        #
        #     I_{i,r}(x) = w1 * I_{i,r-1}(x)  +  w2 * I_{i+1,r-1}(x)
        #
        # where
        #
        #     w1 = (x - t_i) / (t_{i+r}   - t_i)
        #     w2 = (t_{i+r+1} - x) / (t_{i+r+1} - t_{i+1})
        #
        # Vectorization strategy:
        #   - I has shape (N,Q) for degree r-1
        #   - I_ip1 is I shifted by one index
        # ------------------------------------------------------
        for r in range(1, p+1):
            # denominators of the weights for degree r
            den1 = (self.t[r:r+self.N] - self.t[:self.N]).unsqueeze(1)          # (N,1)
            den2 = (self.t[r+1:r+1+self.N] - self.t[1:self.N+1]).unsqueeze(1)   # (N,1)

            # Shift basis upward in i-index: I_{i+1, r-1}
            # Last row is zero (no basis above index N-1)
            I_ip1 = torch.cat([I[1:], torch.zeros_like(I[:1])], dim=0)

            # Weights of the Cox-de Boor recursion with safe division
            w1 = torch.where(den1 > 0, (x - self.t[:self.N].unsqueeze(1)) / den1, torch.zeros_like(I))
            w2 = torch.where(den2 > 0, (self.t[r+1:r+1+self.N].unsqueeze(1) - x) / den2, torch.zeros_like(I))

            # Apply Cox–de Boor recurrence
            I = w1 * I + w2 * I_ip1

        # return basis of degree p
        return I

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B-spline basis functions I_{i,p}(x) in parallel.

        Args:
            x: 1D tensor of points, shape (Q,).

        Returns:
            (B=1, N, Q) tensor with I_{i,p}(x).
        """
        # Scale x to parent domain
        x = (x - self.Dx) / self.L
        # Compute basis functions
        output = self._basis_dp(x, self.p)
        # Add batch dimension
        output = output[None,:,:]
        return output
    
    def derivative(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Evaluate n-th derivatives d^n/dx^n I_{i,p}(x) for all i in parallel.

        Args:
            x: 1D tensor of points, shape (Q,).
            n: Order of the derivative.

        Returns:
            (B, N, Q) tensor with d^n/dx^n I_{i,p}(x).
        
        Notes:
            Uses the known B-spline derivative formula:
            d/dx I_{i,p}(x)
            = p/(t_{i+p} - t_i) * I_{i,p-1}(x)
            - p/(t_{i+p+1} - t_{i+1}) * I_{i+1,p-1}(x)
        """
        # Scale x to parent domain
        x = (x - self.Dx) / self.L

        # Select starting basis functions I_{*, p-n}
        dI = self._basis_dp(x, self.p - n)
        # Set polynomial degree for first iteration (order of starting basis functions + 1)
        p_temp = self.p - n + 1
        # Iteratively apply derivative formula n times on starting basis functions
        for k in range(n):
            # Compute weights
            den1 = (self.t[p_temp:p_temp+self.N] - self.t[:self.N]).unsqueeze(1)
            den2 = (self.t[p_temp+1:p_temp+1+self.N] - self.t[1:self.N+1]).unsqueeze(1)
            w1 = torch.where(den1 > 0, p_temp / den1, torch.zeros_like(dI))
            w2 = torch.where(den2 > 0, p_temp / den2, torch.zeros_like(dI))
            # Shift basis upward in i-index: I_{i+1, p_temp-1}
            dI_ip1 = torch.cat([dI[1:], torch.zeros_like(dI[:1])], dim=0)
            # Update derivative
            dI = w1 * dI - w2 * dI_ip1
            # Increase temporary polynomial degree
            p_temp += 1
        # Final derivative
        output = dI
        # Add batch dimension
        output = output[None,:,:]
        # Scale derivative back to original domain
        output = output / (self.L ** n)
        return output
    
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate derivatives d/dx I_{i,p}(x) for all i in parallel.

        Args:
            x: 1D tensor of points, shape (Q,).

        Returns:
            (B, N, Q) tensor with d/dx I_{i,p}(x).
        """
        return self.derivative(x, n=1)  

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate second derivatives d^2/dx^2 I_{i,p}(x) for all i in parallel.

        Args:
            x: 1D tensor of points, shape (Q,).

        Returns:
            (B, N, Q) tensor with d^2/dx^2 I_{i,p}(x).
        """
        return self.derivative(x, n=2)



# Class created to be able to make sure that the constant piecewise basis is accurate
class PiecewiseConstantBasis:
    def __init__(self, N, dtype=torch.float32, device='cpu'):
        self.N = N
        self.dtype = dtype
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 1D tensor of points, shape (Q,). 
               (Expected to be normalized between 0 and 1).
        Returns:
            (B=1, N, Q) tensor with basis evaluations.
        """
        # Map normalized grid to bin indices
        indices = torch.floor(x * self.N).long()
        indices = torch.clamp(indices, 0, self.N - 1)
        
        # Create the (Q, N) projection matrix
        B = torch.zeros((len(x), self.N), dtype=self.dtype, device=self.device)
        B.scatter_(1, indices.unsqueeze(1), 1.0)
        
        # Transpose to (N, Q) and add batch dimension to return (1, N, Q)
        return B.T.unsqueeze(0)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        # The gradient of a piecewise constant basis is 0 almost everywhere
        return torch.zeros((1, self.N, len(x)), dtype=self.dtype, device=self.device)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        # The laplacian of a piecewise constant basis is 0 almost everywhere
        return torch.zeros((1, self.N, len(x)), dtype=self.dtype, device=self.device)


# Class taken from Joost Prins, produces the product of basis functions over multiple dimensions
class TensorProductBasis:
    """
    Tensor-product (Kronecker) basis built from multiple 1D basis objects.

    This class combines several 1D basis objects into a d-dimensional basis by
    taking tensor products of the 1D basis functions. It evaluates a field
    given modal coefficients `u_m` at tensor-product evaluation points and
    computes partial derivatives by replacing a 1D basis with its derivative
    in the corresponding dimension.

    Requirements for each 1D basis object:
      - .n attribute: number of basis functions in that 1D basis
      - .forward(xi): returns shape (N_i, Q_i)
      - .grad(xi): returns shape (N_i, Q_i)

    Attributes:
        bases (tuple): Sequence of 1D basis objects.
        d (int): Number of dimensions (len(bases)).
        N (tuple): Number of basis functions per dimension (N_0, ..., N_{d-1}).
    """
    def __init__(self, bases):
        """
        Initialize the tensorized basis.

        Args:
            bases (list): A list of 1D basis objects.
        """
        self.bases = bases
        self.d = len(bases)
        self.N = ()
        for i in range(len(bases)):
            self.N += (bases[i].N,)

    def Phi(self, X):
        """
        Evaluate the 1D basis functions at provided points.
        
        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (Q_i,).
        Returns:
            tuple: Sequence of basis evaluation matrices for each dimension.
                   Each element has shape (N_i, Q_i).
        """
        # Construct basis evaluations
        Phi = ()
        # For each dimension, compute basis evaluation matrix
        for i in range(self.d):
            Phi += (self.bases[i].forward(X[i]),)
        return Phi

    def forward(self, U_m, X):
        """
        Evaluate the tensor-product expansion at provided points.

        Args:
            U_m (torch.tensor): Modal coefficients with shape (B, prod(N)) or
                              (B, N_1, N_2, ..., N_d). B is batch size.
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (Q_i,).

        Returns:
            torch.tensor: Evaluated values at tensor-product grid points,
                        shape (B, Q_i_0, ..., Q_i_{d-1}).
        """
        # reshape coefficients into (B, N_0, N_1, ..., N_{d-1})
        u_q = U_m.reshape(U_m.shape[0],*self.N)
        # sequentially contract coefficient tensor with each 1D basis evaluation
        for i in range(self.d):
            # move basis axis to the end for einsum
            u_q = u_q.moveaxis(1, -1) 
            # Compute basis evaluation matrix in dimension i
            Phi_i = self.bases[i].forward(X[i])
            # Contract along i-th dimension
            u_q = opt_einsum.contract('B...n,Bnq->B...q', u_q, Phi_i)
        return u_q

    def grad(self, U_m, X):
        """
        Compute partial derivatives of the tensor-product expansion.
        For each spatial dimension i the method replaces the 1D basis eval in
        that dimension with its derivative and contracts as in `forward`.

        Args:
            U_m (torch.tensor): Modal coefficients, shape (B, prod(n)) or (B, N_0, ..., N_{d-1}).
            X (tuple/list): Sequence of point arrays for each dimension.
                            x[i] has shape (Q_i,).
        Returns:
            torch.tensor: Gradients at tensor-product points with shape
                        (B, Q_i_0, ..., Q_i_{d-1}, d). The last axis indexes
                        partial derivatives w.r.t. each coordinate.
        """
        # Compute derivative for each dimension by contracting with dPhi in that dim.
        gradu_q = ()
        for i in range(self.d):
            # start from coefficients reshaped to (B, N_0, ..., N_{d-1})
            u_q = U_m.reshape(U_m.shape[0],*self.N)
            for j in range(self.d):
                u_q = u_q.moveaxis(1, -1)  # move basis axis to the end for einsum
                if j==i:
                    # use derivative matrix in the i-th dimension
                    dPhi_j = self.bases[j].grad(X[j])
                    # contract along j-th dimension
                    u_q = opt_einsum.contract('B...n,Bnq->B...q', u_q, dPhi_j)
                else:
                    # use value matrix in other dimensions
                    Phi_j = self.bases[j].forward(X[j])
                    # contract along j-th dimension
                    u_q = opt_einsum.contract('B...n,Bnq->B...q', u_q, Phi_j)
            # append gradient in direction i to gradu_q tuple
            gradu_q += (u_q,)
        #Reshape such that the last axis indexes the coordinate to which derivative has been taken
        gradu_q = torch.stack(gradu_q, axis=-1)
        return gradu_q

    def laplacian(self, U_m, X):
        """
        Compute Laplacian of the tensor-product expansion.
        For each spatial dimension i the method replaces the 1D basis eval in
        that dimension with its second derivative and contracts as in `forward`.

        Args:
            U_m (torch.tensor): Modal coefficients, shape (B, prod(N)) or (B, N_0, ..., N_{d-1}).
            X (tuple/list): Sequence of point arrays for each dimension.
                            x[i] has shape (Q_i,).
        Returns:
            torch.tensor: Laplacians at tensor-product points with shape
                        (B, Q_i_0, ..., Q_i_{d-1}, d). The last axis indexes
                        second derivatives w.r.t. each coordinate.
        """
        # Compute derivative for each dimension by contracting with ddPhi in that dim.
        lapu_q = ()
        for i in range(self.d):
            # start from coefficients reshaped to (B, N_0, ..., N_{d-1})
            U_q = U_m.reshape(U_m.shape[0],*self.N)
            for j in range(self.d):
                U_q = U_q.moveaxis(1, -1)  # move basis axis to the end for einsum
                if j==i:
                    # use second derivative matrix in the i-th dimension
                    ddPhi_j = self.bases[j].laplacian(X[j])
                    # contract along j-th dimension
                    U_q = opt_einsum.contract('B...n,Bnq->B...q', U_q, ddPhi_j)
                else:
                    # use value matrix in other dimensions
                    Phi_j = self.bases[j].forward(X[j])
                    # contract along j-th dimension
                    U_q = opt_einsum.contract('B...n,Bnq->B...q', U_q, Phi_j)
            # append gradient in direction i to gradu_q tuple
            lapu_q += (U_q,)
        #Reshape such that the last axis indexes the coordinate to which derivative has been taken
        lapu_q = torch.stack(lapu_q, axis=-1)
        return lapu_q

    def integrate(self, U_m, W_q, X_q):
        """
        Integrate the tensor-product expansion over the domain using
        provided quadrature weights and points.

        Args:
            U_m (torch.tensor): Modal coefficients with shape (B, prod(N)) or
                              (B, N_1, N_2, ..., N_d). B is batch size.
            W_q (tuple/list): Sequence of quadrature weight arrays for each dimension.
                              W_q[i] has shape (Q_i,).
            X_q (tuple/list): Sequence of quadrature point arrays for each dimension.
                              X_q[i] has shape (Q_i,).

        Returns:
            torch.tensor: Integrated values over the domain, shape (B,).
        """
        integral = U_m
        # Sequentially contract with basis evaluations and quadrature weights
        for i in range(self.d):
            Phi_i = self.bases[i].forward(X_q[i])  # (B, N_i, Q_i)
            W_i = W_q[i]  # (Q_i,)
            integral_i = opt_einsum.contract('q,Bnq->Bn', W_i, Phi_i)  # (B, N_i)
            # Combine basis evaluation and weights
            integral = opt_einsum.contract('Bn...,Bn->B...', integral, integral_i)
            # integral = integral.moveaxis(1, -1)  # move basis axis to the end for next einsum
        # After all contractions, integral has shape (B,)
        return integral
    
    def integrals(self, W_q, X_q):
        """
        Integrate the tensor-product expansion over the domain using
        provided quadrature weights and points.

        Args:
            U_m (torch.tensor): Modal coefficients with shape (B, prod(N)) or
                              (B, N_1, N_2, ..., N_d). B is batch size.
            W_q (tuple/list): Sequence of quadrature weight arrays for each dimension.
                              W_q[i] has shape (Q_i,).
            X_q (tuple/list): Sequence of quadrature point arrays for each dimension.
                              X_q[i] has shape (Q_i,).

        Returns:
            torch.tensor: Integrated values over the domain, shape (B,N_0,...,N_{d-1}).
        """
        Phi_0 = self.bases[0].forward(X_q[0])
        integrals = opt_einsum.contract('q,Bnq->Bn', W_q[0], Phi_0)  # (B, N_0)
        # Sequentially contract with basis evaluations and quadrature weights
        for i in range(1,self.d):
            Phi_i = self.bases[i].forward(X_q[i])  # (B, N_i, Q_i)
            W_i = W_q[i]  # (Q_i,)
            integral_i = opt_einsum.contract('q,Bnq->Bn', W_i, Phi_i)  # (B, N_i)
            # Combine basis evaluation and weights
            integrals = opt_einsum.contract('Bn...,Bm->Bnm...', integrals, integral_i)
            # integrals = integrals.moveaxis(1, -1)  # move basis axis to the end for next einsum
        # After all contractions, integral has shape (B,)
        return integrals
    

# Creates a piecewise constant basis matrix such that multiple dimensions are dealt with, in this case basis_t*basis_v
def make_basis_matrix(Nt, Nv, N_basis_t, N_basis_v):
    t_grid = torch.linspace(0.0, 10.0, Nt, dtype=torch.float32, device='cpu')
    v_grid = torch.linspace(-4.0, 4.0, Nv, dtype=torch.float32, device='cpu')

    K = N_basis_t * N_basis_v 
    # basis_t = BSplineBasis(N=N_basis_t, p=3, C=2, dtype=torch.float32, device='cpu')
    # basis_v = BSplineBasis(N=N_basis_v, p=3, C=2, dtype=torch.float32, device='cpu')
    basis_t = PiecewiseConstantBasis(N=N_basis_t, dtype=torch.float32, device='cpu')
    basis_v = PiecewiseConstantBasis(N=N_basis_v, dtype=torch.float32, device='cpu')

    t_norm = (t_grid - t_grid.min()) / (t_grid.max() - t_grid.min())
    v_norm = (v_grid - v_grid.min()) / (v_grid.max() - v_grid.min())

    basis_tp = TensorProductBasis([basis_t, basis_v])
    X_eval = [t_norm, v_norm]

    K = N_basis_t * N_basis_v
    identity_coeffs = torch.eye(K) 

    basis_2d_tensor = basis_tp.forward(identity_coeffs, X_eval)

    basis_matrix = basis_2d_tensor.reshape(K, Nt * Nv)

    return basis_matrix

def make_basis_matrix_with_seperate(Nt, Nv, N_basis_t, N_basis_v):
    t_grid = torch.linspace(0.0, 10.0, Nt, dtype=torch.float32, device='cpu')
    v_grid = torch.linspace(-4.0, 4.0, Nv, dtype=torch.float32, device='cpu')

    K = N_basis_t * N_basis_v 
    basis_t = BSplineBasis(N=N_basis_t, p=3, C=2, dtype=torch.float32, device='cpu')
    basis_v = BSplineBasis(N=N_basis_v, p=3, C=2, dtype=torch.float32, device='cpu')

    t_norm = (t_grid - t_grid.min()) / (t_grid.max() - t_grid.min())
    v_norm = (v_grid - v_grid.min()) / (v_grid.max() - v_grid.min())

    basis_tp = TensorProductBasis([basis_t, basis_v])
    X_eval = [t_norm, v_norm]

    K = N_basis_t * N_basis_v
    identity_coeffs = torch.eye(K) 

    basis_2d_tensor = basis_tp.forward(identity_coeffs, X_eval)

    basis_matrix = basis_2d_tensor.reshape(K, Nt * Nv)

    return basis_matrix, basis_t, basis_v


# Class to also make sure the piecewise linear basises are correct
class PiecewiseLinearBasis:
    """
    Discontinuous (C^{-1}) piecewise linear basis.

    Each of the N bins gets exactly 2 independent basis functions:
      - phi_{2i}  (x) : the "left"  hat  that is 1 at the left  edge of bin i and 0 at the right
      - phi_{2i+1}(x) : the "right" hat  that is 0 at the left  edge of bin i and 1 at the right

    This gives N_basis = 2 * N total basis functions and zero continuity
    across bin boundaries (C^{-1}), matching the BSplineBasis(p=1, C=-1)
    used in the projection script.

    The basis is evaluated at normalised coordinates in [0, 1].
    Internally each bin i covers [i/N, (i+1)/N].

    Interface mirrors PiecewiseConstantBasis and BSplineBasis so it can be
    dropped into TensorProductBasis without any other changes.

    Attributes
    ----------
    N       : int            number of bins
    N_basis : int            total number of basis functions (= 2 * N)
    dtype   : torch.dtype
    device  : torch.device
    """

    def __init__(self, N: int, dtype=torch.float32, device="cpu"):
        self.N       = 2 * N          # expose 2N so TensorProductBasis sees the right count
        self._N_bins = N              # keep the original bin count for internal arithmetic
        self.dtype   = dtype
        self.device  = device

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all 2*N basis functions at the given points.

        Parameters
        ----------
        x : (Q,) tensor
            Normalised evaluation points in [0, 1].

        Returns
        -------
        (1, 2*N, Q) tensor
            Row 2i   -> left  linear ramp in bin i   (1 at left edge,  0 at right)
            Row 2i+1 -> right linear ramp in bin i   (0 at left edge,  1 at right)
        """
        N   = self._N_bins
        Q   = x.shape[0]
        out = torch.zeros((2 * N, Q), dtype=self.dtype, device=self.device)

        # Bin edges in normalised coords
        left_edges  = torch.arange(N, dtype=self.dtype, device=self.device) / N   # (N,)
        right_edges = left_edges + 1.0 / N                                         # (N,)

        # Local coordinate inside each bin: xi in [0, 1]
        # x shape (Q,), left_edges shape (N,) -> broadcast to (N, Q)
        xi = (x.unsqueeze(0) - left_edges.unsqueeze(1)) * N   # (N, Q)

        # Indicator: point x lies inside bin i
        in_bin = (x.unsqueeze(0) >= left_edges.unsqueeze(1)) & \
                 (x.unsqueeze(0) <  right_edges.unsqueeze(1))  # (N, Q) bool

        # Handle right endpoint: clamp into last bin
        at_right = (x == 1.0).unsqueeze(0)                     # (1, Q) -> broadcast
        in_bin[-1] = in_bin[-1] | at_right.squeeze(0)
        xi[-1]     = torch.where(at_right.squeeze(0), torch.ones_like(xi[-1]), xi[-1])

        # Left ramp  phi_{2i}  = 1 - xi   (1 at left, 0 at right)
        # Right ramp phi_{2i+1}= xi        (0 at left, 1 at right)
        left_ramp  = (1.0 - xi) * in_bin   # (N, Q)
        right_ramp = xi         * in_bin   # (N, Q)

        # Interleave: rows [0,2,4,...] = left ramps, [1,3,5,...] = right ramps
        out[0::2] = left_ramp
        out[1::2] = right_ramp

        return out.unsqueeze(0)   # (1, 2*N, Q)

    # ------------------------------------------------------------------
    # Derivatives  (piecewise constant within each bin, 0 at boundaries)
    # ------------------------------------------------------------------
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        First derivative of each basis function.

        Within bin i:
          d/dx phi_{2i}   = -N   (slope of the left  ramp in normalised coords)
          d/dx phi_{2i+1} = +N   (slope of the right ramp in normalised coords)
        At bin boundaries the derivative is zero (discontinuous basis).

        Returns
        -------
        (1, 2*N, Q) tensor
        """
        N   = self._N_bins
        Q   = x.shape[0]
        out = torch.zeros((2 * N, Q), dtype=self.dtype, device=self.device)

        left_edges  = torch.arange(N, dtype=self.dtype, device=self.device) / N
        right_edges = left_edges + 1.0 / N

        in_bin = (x.unsqueeze(0) >= left_edges.unsqueeze(1)) & \
                 (x.unsqueeze(0) <  right_edges.unsqueeze(1))
        in_bin[-1] = in_bin[-1] | (x == 1.0)

        # d/dx (1 - xi) = -N,  d/dx xi = +N   (chain rule through xi = (x - left)*N)
        out[0::2] = -float(N) * in_bin   # left  ramp derivatives
        out[1::2] = +float(N) * in_bin   # right ramp derivatives

        return out.unsqueeze(0)   # (1, 2*N, Q)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Second derivative — identically zero for piecewise linear functions.

        Returns
        -------
        (1, 2*N, Q) zero tensor
        """
        return torch.zeros(
            (1, 2 * self._N_bins, x.shape[0]),
            dtype=self.dtype, device=self.device
        )


# Simple class to again also make a piecewises linear basis matrix
def make_basis_matrix_piecewise_linear(Nt, Nv, N_basis_t, N_basis_v,time):

    assert N_basis_t % 2 == 0, "N_basis_t must be even for C^{-1} piecewise linear"
    assert N_basis_v % 2 == 0, "N_basis_v must be even for C^{-1} piecewise linear"

    # Physical grids
    t_grid = torch.linspace(0.0, time, Nt, dtype=torch.float32)
    v_grid = torch.linspace(-4.0,  4.0, Nv, dtype=torch.float32)

    # Normalise to [0, 1] — PiecewiseLinearBasis works in normalised coords
    t_norm = (t_grid - t_grid.min()) / (t_grid.max() - t_grid.min())
    v_norm = (v_grid - v_grid.min()) / (v_grid.max() - v_grid.min())

    # 1D bases: N_bins = N_basis // 2  so that total DOF = 2 * N_bins = N_basis
    basis_t = PiecewiseLinearBasis(N=N_basis_t // 2, dtype=torch.float32)
    basis_v = PiecewiseLinearBasis(N=N_basis_v // 2, dtype=torch.float32)

    # Tensor-product basis
    basis_tp = TensorProductBasis([basis_t, basis_v])
    X_eval   = [t_norm, v_norm]

    # Identity-coefficient trick: evaluate each basis function independently
    K               = N_basis_t * N_basis_v
    identity_coeffs = torch.eye(K)                          # (K, K)

    basis_2d_tensor = basis_tp.forward(identity_coeffs, X_eval)   # (K, Nt, Nv)
    basis_matrix    = basis_2d_tensor.reshape(K, Nt * Nv)          # (K, Nt*Nv)

    return basis_matrix, basis_t, basis_v