import numpy as np
import torch
import gpytorch
from scipy.stats import qmc


import math

class EntropySelectionStrategy:
    '''Class for Active Learning Selection (by Irina) or random selection'''
    def __init__(self, blur=0.15, beta=50, tolerance_sampling = 1.0, proximity_sampling = 0.1):
        self.blur = blur
        self.beta = beta
        self.tolerance_sampling = tolerance_sampling
        self.proximity_sampling = proximity_sampling

    def best_not_yet_chosen(self, score, previous_indices, device):
        candidates = torch.sort(score, descending=True)[1].to(device)
        for next_index in candidates:
            if int(next_index) not in previous_indices:
                return next_index

    def gibbs_sample(self, score, beta, device):
        '''Chooses next point based on probability that a random value is smaller than the cumulative sum 
            of the probabilites. These are calculated with the smoothed batch entropy
            - if beta is high: deterministic selection with only highest entropies
            - if beta is low: more random selection with each point having a similar probility
        '''
        probs = torch.exp(beta * (score - torch.max(score))).to(device)
        probs /= torch.sum(probs)
        cums = torch.cumsum(probs, dim=0)
        rand = torch.rand(size=(1,)).to(device)[0]
        return torch.sum(cums < rand)

    def approximate_batch_entropy(self, mean, cov, device):

        mean = mean.float()
        cov = cov.float()

        n = mean.shape[-1]
        d = torch.diag_embed(1. / mean).to(device)
        x = d @ cov @ d
        I = torch.eye(n)[None, :, :].to(device)
        return (torch.logdet(x + I) - torch.logdet(x + 2 * I) + n * torch.log(torch.tensor(2.0))) / torch.log(torch.tensor(2.0))

    def smoothed_batch_entropy(self, blur, device):
        return lambda mean, cov: self.approximate_batch_entropy(mean + blur * torch.sign(mean).to(device), cov, device)

    def iterative_batch_selector(self, score_function, choice_function, gp_mean, gp_covar, N, device):
        '''Chooses the next points iterativley. The point with maximum entropy is always chosen first, 
            then the next indices are selected with the choice function - gibbs_sampling or best_not_yet_chosen
            The covariance matrix and the mean vector are updated iterativly, based on the already chosen points
        '''

        score = score_function(gp_mean[:, None], torch.diag(gp_covar)[:, None, None]).to(device)
        first_index = torch.argmax(score).to(device)
        indices = [int(first_index)]

        num_pts = len(gp_mean)

        for _ in range(N - 1):
            center_cov = torch.stack([gp_covar[indices, :][:, indices]] * num_pts).to(device)
            side_cov = gp_covar[:, None, indices].to(device)
            bottom_cov = gp_covar[:, indices, None].to(device)
            end_cov = torch.diag(gp_covar)[:, None, None].to(device)

            cov_batch = torch.cat([
                torch.cat([center_cov, side_cov], axis=1),
                torch.cat([bottom_cov, end_cov], axis=1),
            ], axis=2)

            center_mean = torch.stack([gp_mean[indices]] * num_pts).to(device)
            new_mean = gp_mean[:, None].to(device)
            mean_batch = torch.cat([center_mean, new_mean], axis=1)

            score = score_function(mean_batch, cov_batch).to(device)
            next_index = choice_function(score, indices)
            indices.append(int(next_index))

        return indices


    def select_new_points(self, pipeline, N=4):
        model = pipeline.model
        likelihood = pipeline.likelihood
        device = pipeline.device
        n_dim = pipeline.n_dim
        thr = torch.tensor([pipeline.thr], device=device)
        num_samples = 1 if not pipeline.is_deep else pipeline.num_samples

        # Size of end pool
        # For DKL the covariance evaluation is more expensive
        if pipeline.use_dkl:
            n_pool = 1000 
        else:
            n_pool = 10000
        if self.tolerance_sampling != 0:
            n_large = 1000000  # Size of the initial pool

            # Sample out of the inital pool with LHS
            x_large = torch.tensor(
                qmc.LatinHypercube(d=n_dim).random(n=n_large),
                dtype=torch.float32
            ).to(device)

            model.eval()
            likelihood.eval()

            batch_size = 100_000
            means, vars = [], []

            print("Initial Pool selection")
            for i in range(0, len(x_large), batch_size):
                x_batch = x_large[i:i+batch_size]
                with torch.no_grad(), \
                    gpytorch.settings.eval_cg_tolerance(1e-4), \
                    gpytorch.settings.max_cg_iterations(300), \
                    gpytorch.settings.fast_pred_var(False), \
                    gpytorch.settings.fast_pred_samples(True), \
                    gpytorch.settings.cholesky_jitter(pipeline.jitter), \
                    gpytorch.settings.num_likelihood_samples(num_samples):

                    preds = likelihood(model(x_batch))
                    mean = preds.mean.detach()
                    var = preds.variance.detach()

                    if pipeline.is_deep:
                        means.append(mean.mean(dim=0).squeeze())
                        vars.append(var.mean(dim=0))
                    else:
                        means.append(mean)
                        vars.append(var)

            mean = torch.cat(means)
            var = torch.cat(vars)

            # Focus on points near threshold 
            mask = (mean > thr - self.tolerance_sampling) & (mean < thr + self.tolerance_sampling)
            candidates = x_large[mask]
            # Choose pool also based on entropy = distance * variance
            if self.proximity_sampling != 0:
                proximity = torch.exp(-((mean[mask] - thr) ** 2) / self.proximity_sampling)
                entropy_score = proximity * var[mask]

                topk = torch.topk(entropy_score, k=n_pool, largest=True)
                x_pool = x_large[mask][topk.indices]
            else:
                if len(candidates) > n_pool:
                    idx = torch.randperm(len(candidates), device=device)[:n_pool]
                    x_pool = candidates[idx]
                else:
                    x_pool = candidates
                    print("[INFO] Candidates selected")
                #x_pool = x_large[mask]
        else:
            x_pool = torch.tensor(
                qmc.LatinHypercube(d=pipeline.n_dim).random(n=n_pool),
                dtype=torch.float32
            ).to(device)
        
        if pipeline.target == "CLs":
            x_pool = pipeline.x_train

        if pipeline.evaluation_mode:
            # Set eval mode for model and likelihood
            model.eval()
            likelihood.eval()

        print("[INFO] Focused pool selection")
        # Calculate gp-mean and covariance out of the pool
        with torch.no_grad(), \
            gpytorch.settings.eval_cg_tolerance(1e-4), \
            gpytorch.settings.max_cg_iterations(300), \
            gpytorch.settings.fast_pred_var(False), \
            gpytorch.settings.fast_pred_samples(True), \
            gpytorch.settings.cholesky_jitter(pipeline.jitter), \
            gpytorch.settings.num_likelihood_samples(num_samples):

            preds_focus = likelihood(model(x_pool))

            # Take the mean of the samples predicted by the Deep GP
            if pipeline.is_deep:
                mean = preds_focus.mean.detach().mean(axis=0).squeeze()
                covar = preds_focus.covariance_matrix.detach().mean(dim=0)
            # Exact GP has only one sample
            else:
                mean = preds_focus.mean.detach()
                covar = preds_focus.covariance_matrix.detach()

        # Select entropy based new points
        score_function = self.smoothed_batch_entropy(blur=self.blur, device=device)
        choice_function = lambda score, indices: self.gibbs_sample(score, self.beta, device)

        print("[DEBUG] Choice done")

        selected_indices = self.iterative_batch_selector(score_function, choice_function, mean - thr, covar, N, device)

        # Extract means und covariances of the selected points
        sel_mean = mean[selected_indices]                
        sel_covar = covar[selected_indices][:, selected_indices] 
        print("[INFO] Mean of the selected points:", sel_mean)
        print("[INFO] Coveriance of the selected points:", sel_covar)
        
        # Calculate the entropy per point
        per_point_entropy = []
        for i in range(sel_mean.shape[0]):
            m1 = sel_mean[i].view(1, 1)           
            c1 = sel_covar[i, i].view(1, 1, 1)     
            s1 = score_function(m1, c1).item()
            per_point_entropy.append(s1)
        print("[INFO] Entropy of the selected points:", per_point_entropy)

        return x_pool[list(selected_indices)]

    def select_randomly(self, pipeline, N=4):
        '''Function to create random points with a fresh, non-seeded generator'''
        rng = np.random.default_rng() 
        new_points = rng.random((N, pipeline.n_dim))
        return torch.tensor(new_points, dtype=torch.float32, device=pipeline.device)


