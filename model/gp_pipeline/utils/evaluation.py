import torch
import os
import numpy as np
import gpytorch
import pandas as pd

def compute_accuracy(predictions, truths, threshold):
    '''Function to compute the accuracy and the confusion matrix'''
    # Convert to torch
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(truths, np.ndarray):
        truths = torch.tensor(truths)

    preds = predictions.squeeze()
    truths = truths.squeeze()
    assert preds.shape == truths.shape, "[WARNING] Prediction and truth tensor shapes do not match."

    # Compute true positives(TP), false positives(FP), true negatives(TN) and false negatives(FN)
    TP = ((preds > threshold) & (truths > threshold)).sum().item()
    FP = ((preds > threshold) & (truths < threshold)).sum().item()
    FN = ((preds < threshold) & (truths > threshold)).sum().item()
    TN = ((preds < threshold) & (truths < threshold)).sum().item()
    
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0.0
    conf_matrix = np.array([[TN, FP], [FN, TP]])

    return acc, conf_matrix

def compute_gof_metrics(predictions, truths, up, low, thr):
    '''Function to compute regression GoF metrics'''
    mean  = predictions.squeeze()
    true  = truths.squeeze()
    upper = up.squeeze()
    lower = low.squeeze()

    eps = torch.finfo(mean.dtype).eps
    span = (upper - lower).abs().clamp_min(eps)

    # === Unweighted ===
    # Mean_squared 
    mse = torch.mean((mean - true) ** 2)
    rmse = torch.sqrt(mse)

    # R_squared - explains how much of variance of data is explained by model
    ss_res = torch.sum((mean - true) ** 2)
    ss_tot = torch.sum((true - torch.mean(true)) ** 2).clamp_min(eps)
    r2 = 1.0 - ss_res / ss_tot

    # Pull and Chi_squared
    pull = (mean - true) / span
    chi2 = torch.sum(pull ** 2)
    dof  = max(int(true.numel()) - 1, 1)
    chi2_red = chi2 / dof
    
    abs_pull = pull.abs()
    mean_abs_pull = torch.mean(abs_pull)
    rms_pull = torch.sqrt(torch.mean(abs_pull ** 2))

    # ==== Weighted ====
    # Weighting based on distance to threshold 
    distance = torch.abs(truths - thr)
    weights = torch.exp(-2.0 * distance)
    
    mse_w  = torch.mean(weights * (mean - true) ** 2)
    rmse_w = torch.sqrt(mse_w)

    ss_res_w = torch.sum(weights * (mean - true) ** 2)
    ss_tot_w = torch.sum(weights * (true - torch.mean(true)) ** 2).clamp_min(eps)
    r2_w     = 1.0 - ss_res_w / ss_tot_w

    pull_w = pull
    chi2_w = torch.sum(weights * pull_w ** 2)
    chi2_red_w = chi2_w / dof

    mean_abs_pull_w = torch.mean(weights * abs_pull)
    rms_pull_w = torch.sqrt(torch.mean(weights * (abs_pull ** 2)))

    return {
        "mse": mse.item(), "rmse": rmse.item(), "r2": r2.item(),
        "chi2": chi2.item(), "chi2_red": chi2_red.item(),
        "mean_abs_pull": mean_abs_pull.item(), "rms_pull": rms_pull.item(),
        "mse_w": mse_w.item(), "rmse_w": rmse_w.item(), "r2_w": r2_w.item(),
        "chi2_w": chi2_w.item(), "chi2_red_w": chi2_red_w.item(),
        "mean_abs_pull_w": mean_abs_pull_w.item(), "rms_pull_w": rms_pull_w.item(),
    }


def misclassified(predictions, truths, thr):
    '''Function to determine the misclassified points'''
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(truths, np.ndarray):
        truths = torch.tensor(truths)

    preds = predictions.squeeze()
    truths = truths.squeeze()

    assert preds.shape == truths.shape, "[WARNING] Prediction and truth tensor shapes do not match."

    # Find false positives(FP) and false negatives(FN)
    pred_pos  = preds >= thr
    truth_pos = truths >= thr
    FP = pred_pos & ~truth_pos
    FN = ~pred_pos &  truth_pos

    # Determine their index and concatenate them
    fp_idx = torch.where(FP)[0]
    fn_idx = torch.where(FN)[0]
    idx = torch.cat([fp_idx, fn_idx]) 
    print(f"idx: {idx}")

    return idx

def compute_weighted_accuracy(predictions, truths, thr, alpha=2.0):
    '''Function to calculate accuracy weighted for the distance to the threshold'''
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(truths, np.ndarray):
        truths = torch.tensor(truths)

    preds = predictions.squeeze()
    truths = truths.squeeze()

    assert preds.shape == truths.shape, "[WARNING] Prediction and truth tensor shapes do not match."

    pred_pos  = preds >= thr
    truth_pos = truths >= thr

    # Weighting based on distance to threshold 
    distance = torch.abs(truths - thr)
    weights = torch.exp(-alpha * distance)

    TP = ((pred_pos & truth_pos) * weights).sum().item()
    TN = ((~pred_pos & ~truth_pos) * weights).sum().item()
    FP = ((pred_pos & ~truth_pos) * weights).sum().item()
    FN = ((~pred_pos & truth_pos) * weights).sum().item()

    total_weight = TP + TN + FP + FN
    weighted_acc = (TP + TN) / total_weight if total_weight > 0 else 0.0

    # Weighted Confusion Matrix
    conf_matrix = np.array([[TN, FP], [FN, TP]])

    return weighted_acc, conf_matrix


def evaluate_and_log(self, cfg, name, output_dir):
    '''Function to evaluate the accuracy and save it in a csv file'''

    if cfg.target == "Toy":
        x_eval = torch.tensor(self.x_test.cpu().numpy(), dtype=torch.float32).to(self.device)
        y_true = torch.tensor(self.truth_fn(x_eval.cpu().numpy()), dtype=torch.float32).to(self.device)
    else:
        x_eval = torch.tensor(self.x_true.cpu().numpy(), dtype=torch.float32).to(self.device) 
        y_true = torch.tensor(self.y_true.cpu().numpy(), dtype=torch.float32).to(self.device) 
        print(f"[INFO] True Values: {x_eval} and {y_true}")

    self.model.eval()
    self.likelihood.eval()

    num_samples = 1 if not self.is_deep else self.num_samples
    with torch.no_grad(), \
        gpytorch.settings.eval_cg_tolerance(1e-4), \
        gpytorch.settings.max_cg_iterations(300), \
        gpytorch.settings.fast_pred_var(False), \
        gpytorch.settings.fast_pred_samples(True), \
        gpytorch.settings.cholesky_jitter(self.jitter), \
        gpytorch.settings.num_likelihood_samples(num_samples):
        preds = self.likelihood(self.model(x_eval))
    
    # Create mean over Deep GP samples
    if self.is_deep:
        mean = preds.mean.detach().mean(axis=0)
    else:
        mean = preds.mean.detach()

    if not self.is_mlp:
        lower, upper = preds.confidence_region()
        lower = lower.detach()
        upper = upper.detach()

    # Used normalized targets
    if self.y_norm:
        y_true =torch.from_numpy(self.scaler_y.transform(y_true.cpu().numpy().reshape(-1,1)).squeeze()).to(device)
        self.thr = self.scaler_y.transform([[self.thr]])[0][0]
    print(f"threshold: {self.thr}")

    # if self.target == "Toy":
    #     #acc1, acc2 = compute_nblob_accuracy()
    #     acc, conf = compute_accuracy(mean, y_true, self.thr)
    # else:
    acc, conf = compute_accuracy(mean, y_true, self.thr)
    results = {"accuracy": acc}
    for alpha in (1.0, 2.0, 5.0, 10.0):
        acc_w, conf_w = compute_weighted_accuracy(mean, y_true, self.thr, alpha)
        results[f"weighted_acc_alpha_{alpha}"] = acc_w
    print(f"[INFO] [{name}] Accuracy: {acc:.4f}, Confusion Matrix: {conf.tolist()}")

    # Add other regreesion gof metrics
    if not self.is_mlp:
        gof = compute_gof_metrics(
            predictions=mean,
            truths=y_true,
            up=upper,
            low=lower,
            thr=self.thr
        )
        results.update(gof)


    # Create a csv with all accuracies
    csv_path = os.path.join(output_dir, f"gof_{name}.csv")
    df = pd.DataFrame([results])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    print(f"[INFO] Results saved to {csv_path}")
    
    # Add misclassified points
    idx_mis = misclassified(mean, y_true, self.thr)
    self.misclassified_points = x_eval[idx_mis]

    print(f"[INFO] Missclassified points: {self.misclassified_points}")

def compute_nblob_accuracy(self, test=None, csv_path=None):
    for blob in range(self.n_blobs):
        '''
        For each blob, evaluate the model at test points concentrated around the contour for each gaussian blob.
        Get the test_points for each blob from the n_blob function of the create_truth function.
        '''
        # Evaluate model at test points concentrated around the contour for each gaussian blob
        # Sample random points around mean of corresponding blob with covariance matrix
        mean_tensor = torch.tensor(self.mean[blob], dtype=torch.float32).to(self.device)
        cov_tensor = torch.tensor(self.cov[blob], dtype=torch.float32).to(self.device)
        mvn = torch.distributions.MultivariateNormal(mean_tensor, covariance_matrix=cov_tensor/1.5)

        # Take more points for the bigger circle and less for the smaller one
        if blob == 0:
            input_data = mvn.sample((1000,)).float()
        else:
            input_data = mvn.sample((2000,)).float() 

        # Normalize the input data to be in the range [0, 1]
        mask = (input_data >= 0) & (input_data <= 1)
        mask = mask.all(axis=1)
        input_data = input_data[mask]

        # Calculate the true values (log-scaled)
        truth_values = self.truth0(input_data.cpu().numpy())  

        # Convert back to torch tensors
        true = torch.tensor(truth_values, dtype=torch.float32).to(self.device)  

        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(input_data)

        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy().mean(axis=0)

        print("Predictions Shape:", mean.shape)  # Should be (3280,)
        print("Mean all: ", mean.tolist())

        mean = torch.tensor(mean, dtype=torch.float32).to(self.device)

def evaluate_mlp_and_log(self, cfg, name, output_dir, name_suffix="_mlp_with_al"):
    '''Function to evaluate accuracy for using mlp with AL points'''

    if self.y_norm:
        self.thr = self.scaler_y.transform([[self.thr]])[0][0]

    if cfg.target == "Toy":
        x_eval = torch.tensor(self.x_test.cpu().numpy(), dtype=torch.float32).to(self.device)
        y_true = torch.tensor(self.truth_fn(x_eval.cpu().numpy()), dtype=torch.float32).to(self.device)
    else:
        x_eval = torch.tensor(self.x_true.cpu().numpy(), dtype=torch.float32).to(self.device)
        y_true = torch.tensor(self.y_true.cpu().numpy(), dtype=torch.float32).to(self.device)

    # Evaluate MLP without likelihood
    self.model_mlp.eval()
    with torch.no_grad():
        mean = self.model_mlp(x_eval).squeeze()
    if self.y_norm:
        mean = torch.tensor(self.scaler_y.inverse_transform(mean.reshape(-1,1)).squeeze(), dtype=torch.float32).to(self.device)

    acc, conf = compute_accuracy(mean, y_true, self.thr)
    results = {"accuracy": acc}
    for alpha in (1.0, 2.0, 5.0, 10.0):
        acc_w, _ = compute_weighted_accuracy(mean, y_true, self.thr, alpha)
        results[f"weighted_acc_alpha_{alpha}"] = acc_w

    print(f"[{name}{name_suffix}] Accuracy: {acc:.4f}, Confusion Matrix: {conf.tolist()}")
    csv_path = os.path.join(output_dir, f"gof_{name}{name_suffix}.csv")
    pd.DataFrame([results]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
