"""
Fitting tools for loss scaling law and lr/bs scaling law
"""
import random
import math
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Union
from functools import partial
from multiprocessing import Pool, cpu_count
import warnings

import tqdm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
import scipy.optimize as scipy_opt
from scipy.stats import gaussian_kde

import typer
app = typer.Typer()

# for loss scaling law
class GeneralModel(nn.Module):
    """
    general model class for fitting
        __init__ takes in a List[Dict], each Dict containing the parameter for the model
            Dict is like {"name":"A", "type":"parameter","learnable":True, "init_value":0.0}, {"name":"x", "type":"variable"}
        __init__ also takes in a string which is a python valid expression and the involved variable string is either parameter or variable
            this is the functional form of the model, e.g. something like "A/x", "x**A"
    """
    def __init__(self, model_expr:str,model_params:List[Dict]):
        super(GeneralModel, self).__init__()
        self.model_params = model_params
        self.model_expr = model_expr
        # based on model_params setattr for each parameter
        for param in model_params:
            if param["type"] == "parameter":
                # if not learnable, set requires_grad to False
                setattr(self, param["name"], 
                        nn.Parameter(torch.tensor(param["init_value"], requires_grad=param.get("learnable",False))))

    def forward(self,input_dict:Dict[str,torch.Tensor]):
        # input_dict is like {"x":x}
        # based on model_expr, calculate the output
        # use eval to evaluate the expression
        model_expr = self.model_expr
        for key in input_dict:
            model_expr = model_expr.replace(key, f"input_dict['{key}']")
        for param in self.model_params:
            if param["type"] == "parameter":
                model_expr = model_expr.replace(param["name"], f"self.{param['name']}")
        res = eval(model_expr)
        return res
    
    def __repr__(self):
        # substitute the parameter value with the actual value
        model_expr = self.model_expr
        for param in self.model_params:
            if param["type"] == "parameter":
                model_expr = model_expr.replace(param["name"], f"{getattr(self,param['name']).item():.3e}")
        return model_expr

# for loss scaling law
def log_huber_loss_func(pred, target):
    # firt convert both to log space, then apply huber loss
    pred = torch.log(pred)
    target = torch.log(target)
    return nn.functional.huber_loss(pred, target, delta=1e-3)

# util function to find mode
def find_mode(series:pd.Series)->float:
    kde = gaussian_kde(series)
    x_grid = np.linspace(series.min(), series.max(), 1000)
    kde_values = kde(x_grid)
    mode_index = kde_values.argmax()
    mode_value = x_grid[mode_index]
    return mode_value

# construct model from parameter_df
def construct_model_from_parameter_dict(
    parameter_dict:Union[Dict, pd.Series],
    input_keys:List[str], model_expr:str
)->GeneralModel:
    model_params = [
        *[{"name":key, "type":"variable"} for key in input_keys],
        *[
            {"name":name, "type":"parameter", "learnable":True, "init_value":init_value}
            for name, init_value in parameter_dict.items()
        ]
    ]
    model = GeneralModel(model_expr, model_params=model_params)
    return model

# TODO: add more logic for keep val set; one interpoltion, one extrapolation should be best
# TODO: for now, just reserve one extrapolation as val set
def prepare_train_val_data(
    query_data:pd.DataFrame, output_key:str, val_set_size:int=1
)->Tuple[pd.DataFrame, pd.DataFrame]:
    val_data = query_data.nsmallest(val_set_size, output_key)
    train_data = query_data.drop(val_data.index)
    return train_data, val_data

def process_loss_scaling_data(
    df:pd.DataFrame, input_keys:List[str], output_key:str,
)->Tuple[Dict[str,torch.Tensor], torch.Tensor]:
    res = (
        {key:torch.tensor(df[key].values, dtype=torch.float32) for key in input_keys},
        torch.tensor(df[output_key].values, dtype=torch.float32)
    )
    return res

def general_objective_func_and_grad(
        x:np.ndarray, input_dict:Dict[str,torch.Tensor], output:torch.Tensor, 
        model_expr:str, input_keys:List[str], parameter_keys:List[str]
    ) -> Tuple[float, np.ndarray]:
    """
    When actually use, need to pass in the model_expr, input_keys, parameter_keys first
    """
    model = construct_model_from_parameter_dict(
        dict(zip(parameter_keys, x)), input_keys, model_expr
    )
    pred = model(input_dict)
    loss = log_huber_loss_func(pred, output)
    loss.backward()

    grad = np.array([param.grad for name, param in model.named_parameters() if param.requires_grad])

    return loss.item(), grad

def fit_loss_scaling_model(
    data:Tuple[Dict[str,torch.Tensor],torch.Tensor], optimization_method:str,
    parameter_grid:Dict[str,List[float]], objective_func_and_grad:Callable[[np.ndarray, Dict[str, torch.Tensor], torch.Tensor],Tuple[float, np.ndarray]],
    debug:bool=False
)->Dict:
    # always select based on train loss for now
    best_loss = float("inf")
    best_params = None
    best_fit_results = None
    for param_values in itertools.product(
        *[parameter_grid[key] for key in parameter_grid.keys()]
    ):
        x = np.array(param_values)
        result = scipy_opt.minimize(
            objective_func_and_grad,
            x0 = x,
            args=(data[0], data[1]),
            method=optimization_method,
            jac=True
        )
        # calculate loss on target set
        loss = objective_func_and_grad(result.x, data[0], data[1])[0]
        if loss < best_loss:
            best_loss = loss
            best_params = result.x
            best_fit_results = result
            log_dict = {
                "best_loss":best_loss,
                "best_params":best_params,
                "initial_guess":x,
                "train_loss":result.fun,
            }
            if debug:
                print(" | ".join([f"{key}:{val}" for key, val in log_dict.items()]))
    return {
        "best_params":best_params,
        "best_fit_results": best_fit_results,
        "best_loss":best_loss
    }

def fit_bootstrap_loss_scaling_model(
    data_df:pd.DataFrame, optimization_method:str, input_keys:List[str], output_key:str,
    parameter_grid:Dict[str,List[float]], objective_func_and_grad:Callable[[np.ndarray, Dict[str, torch.Tensor], torch.Tensor],Tuple[float, np.ndarray]],
    n_bootstrap:int=2000
)->pd.DataFrame:
    # TODO: add more processing logic
    # According to pair visualization, we found some results are not optimized properly
    with Pool(processes=cpu_count()) as pool:
        fit_results = []
        for _ in tqdm.tqdm(range(n_bootstrap)):
            sample_data_df = data_df.sample(frac=1, replace=True)
            data = process_loss_scaling_data(sample_data_df, input_keys, output_key)
            fit_result = pool.apply_async(fit_loss_scaling_model, (data, optimization_method, parameter_grid, objective_func_and_grad))
            fit_results.append(fit_result)

        bootstrap_results = [fit_result.get() for fit_result in fit_results]
    parameter_keys = list(parameter_grid.keys())
    bootstrap_fit_model_params_df = pd.DataFrame(
        [ result['best_params'] for result in bootstrap_results ],
        columns=parameter_keys
    )

    return bootstrap_fit_model_params_df

def fit_and_eval_bootstrap_loss_scaling_model(
    df:pd.DataFrame, input_keys:List[str], output_key:str, parameter_grid:Dict[str,List[float]],
    n_bootstrap:int=2000,
    seed:int=42
):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_expr = "torch.exp(E) + torch.exp(A)/(N**alpha) + torch.exp(B)/(D**beta)"
    optimization_method="BFGS"

    parameter_keys = list(parameter_grid.keys())
    objective_func_and_grad = partial(
        general_objective_func_and_grad, model_expr=model_expr, input_keys=input_keys, parameter_keys=parameter_keys
    )
    query_data = df.dropna(subset=input_keys + [output_key])
    train_data, val_data = prepare_train_val_data(query_data, output_key)

    bootstrap_fit_model_param_df = fit_bootstrap_loss_scaling_model(
        train_data, optimization_method, input_keys, output_key, 
        parameter_grid, objective_func_and_grad, n_bootstrap=n_bootstrap
    )

    # use seaborn pair to plot the correlation
    # sns.pairplot(bootstrap_fit_model_param_df, diag_kind='kde', plot_kws={'alpha':0.5}, corner=True)
    mode_fit_model_param_df = bootstrap_fit_model_param_df.apply(find_mode)
    model = construct_model_from_parameter_dict(
        mode_fit_model_param_df,input_keys,model_expr, 
    )
    with torch.no_grad():
        val_processed_data = process_loss_scaling_data(val_data, input_keys, output_key)
        model_pred = model(val_processed_data[0])
        val_data[f'pred_{output_key}'] = model_pred.numpy()
        val_data[f'rel_err'] = val_data[f'pred_{output_key}'] / val_data[output_key] - 1

    # only keep data_recipe, input_keys, output_key, pred_output_key, rel_err for print
    show_val_data = val_data[['data_recipe'] + input_keys + [output_key, f'pred_{output_key}', 'rel_err']]

    return {
        "model": model, 
        "param_df": mode_fit_model_param_df, 
        "show_val_data": show_val_data,
        "bootstrap_fit_model_param_df": bootstrap_fit_model_param_df,
    }

# for lr/bs scaling law
def query_optimal_lr_bs_data(
    data:pd.DataFrame, filter_threshold:float=0.0025, 
    seq_len:int=2048,
    group_keys:List[str]=["N", "D"]
)->pd.DataFrame:
    """
    data: pd.DataFrame, columns should contain ["N", "D", "h", "ffnh", "lr", "bs", "ti", "smooth loss"]
    """
    get_M_expr = f"6*(4*h+3*ffnh+2*seq_len)*numl*h"
    # group by N, D; then get minimum "smooth loss" for each group
    grouped = data.groupby(group_keys).agg(min_loss=("smooth loss", "min")).reset_index()
    merged_data = pd.merge(data, grouped, on=group_keys)
    merged_data['relative_error'] = merged_data.eval("`smooth loss`/min_loss - 1")
    merged_data['abs_relative_error'] = np.abs(merged_data['relative_error'])
    # sort by abs_relative_error from smallest to largest
    merged_data = merged_data.sort_values(by='abs_relative_error')
    # calculate log(D), log(N), log(lr), log(bs)
    if "seq_len" not in merged_data.columns:
        merged_data['seq_len'] = seq_len
    merged_data['bs'] = merged_data.eval("bs*seq_len") # change to token level
    merged_data['logD'] = np.log(merged_data['D'])
    merged_data['logN'] = np.log(merged_data['N'])
    merged_data['loglr'] = np.log(merged_data['lr'])
    merged_data['logbs'] = np.log(merged_data['bs'])
    merged_data['logti'] = np.log(merged_data['ti'])
    # also calculate M
    merged_data['M'] = merged_data.eval(get_M_expr)
    merged_data['C'] = merged_data['M'] * merged_data['D']
    merged_data['logM'] = np.log(merged_data['M'])
    merged_data['logC'] = np.log(merged_data['C'])
    query_data = merged_data.query(f"abs_relative_error < {filter_threshold}")
    return query_data

def fit_models(sample_data: pd.DataFrame, X_lr_key: List[str], y_lr_key: str, X_bs_key: List[str], y_bs_key: str):
    X_lr = sample_data[X_lr_key]
    y_lr = sample_data[y_lr_key]
    X_bs = sample_data[X_bs_key]
    y_bs = sample_data[y_bs_key]
    lr_model = LinearRegression().fit(X_lr, y_lr)
    bs_model = LinearRegression().fit(X_bs, y_bs)
    return lr_model, bs_model

def get_bootstrap_lr_bs_scaling_models(
    query_data: pd.DataFrame, n_bootstrap: int = 1000,
    X_lr_key:List[str]=['logN', 'logD'], X_bs_key:List[str]=['logN', 'logD'],
) -> List[LinearRegression]:
    """
    query_data: pd.DataFrame, columns should contain ["logN", "logD", "loglr", "logbs"]
    """
    y_lr_key = 'loglr'
    y_bs_key = 'logbs'

    with Pool(processes=cpu_count()) as pool:
        results = []
        for _ in tqdm.tqdm(range(n_bootstrap)):
            sample_data = query_data.sample(frac=1, replace=True)
            results.append(pool.apply_async(fit_models, (sample_data, X_lr_key, y_lr_key, X_bs_key, y_bs_key)))

        bootstrap_models = [result.get() for result in results]

    return bootstrap_models

def get_lr_bs_scaling_model_parameters(
    bootstrap_models:List[LinearRegression], X_lr_key:List[str]=['logN', 'logD'], X_bs_key:List[str]=['logN', 'logD']
)->pd.DataFrame:
    # get all coefficients, intercept and store in a pandas dataframe
    parameter_df = pd.DataFrame(columns=(
        ['lr_intercept']+[f"lr_coef{key}" for key in [item.replace('log',"") for item in X_lr_key]]+
        ["bs_intercept"]+[f"bs_coef{key}" for key in [item.replace('log',"") for item in X_bs_key]]
    ))
    for lr_model, bs_model in bootstrap_models:
        parameter_df = pd.concat([parameter_df, pd.DataFrame({
                'lr_intercept': [lr_model.intercept_],
                **{
                    f"lr_coef{key}": [lr_model.coef_[i]] for i, key in enumerate([item.replace('log',"") for item in X_lr_key])
                },
                'bs_intercept': [bs_model.intercept_],
                **{
                    f"bs_coef{key}": [bs_model.coef_[i]] for i, key in enumerate([item.replace('log',"") for item in X_bs_key])
                }
            })],
            ignore_index=True
        )
    return parameter_df

def get_mean_lr_bs_scaling_model(
    parameter_df:pd.DataFrame, X_lr_key:List[str]=['logN', 'logD'], X_bs_key:List[str]=['logN', 'logD']
)->Tuple[LinearRegression, LinearRegression]:
    # calculate std and mean for each parameter
    lr_model_mean = LinearRegression()
    lr_model_mean.coef_ = np.array(
        [parameter_df[f'lr_coef{key}'].mean() for key in [item.replace('log',"") for item in X_lr_key]]
    )
    lr_model_mean.intercept_ = parameter_df['lr_intercept'].mean()
    bs_model_mean = LinearRegression()
    bs_model_mean.coef_ = np.array(
        [parameter_df[f'bs_coef{key}'].mean() for key in [item.replace('log',"") for item in X_bs_key]]
    )
    bs_model_mean.intercept_ = parameter_df['bs_intercept'].mean()
    return lr_model_mean, bs_model_mean

def load_bootstrap_lr_bs_models(
    parameter_df:pd.DataFrame, X_lr_key:List[str]=['logN', 'logD'], X_bs_key:List[str]=['logN', 'logD']
)->List[Tuple[LinearRegression, LinearRegression]]:
    bootstrap_models = []
    for i, row in parameter_df.iterrows():
        lr_model = LinearRegression()
        lr_model.coef_ = np.array([row[f'lr_coef{key}'] for key in [item.replace('log',"") for item in X_lr_key]])
        lr_model.intercept_ = row['lr_intercept']
        bs_model = LinearRegression()
        bs_model.coef_ = np.array([row[f'bs_coef{key}'] for key in [item.replace('log',"") for item in X_bs_key]])
        bs_model.intercept_ = row['bs_intercept']
        bootstrap_models.append((lr_model, bs_model))

    return bootstrap_models

def predict_with_model(
    input_data_df: pd.DataFrame, model: Tuple[LinearRegression, LinearRegression], 
    X_lr_key:List[str]=['logN', 'logD'], X_bs_key:List[str]=['logN', 'logD']
) -> pd.DataFrame:
    tmp_output_data_df = input_data_df.copy()
    tmp_output_data_df['loglr'] = model[0].predict(input_data_df[X_lr_key])
    tmp_output_data_df['logbs'] = model[1].predict(input_data_df[X_bs_key])
    return tmp_output_data_df

def get_lr_bs_pred(
    input_data_df: pd.DataFrame,
    bootstrap_models: List[Tuple[LinearRegression, LinearRegression]],
    mean_models: Tuple[LinearRegression, LinearRegression],
    X_lr_key:List[str]=['logN', 'logD'], X_bs_key:List[str]=['logN', 'logD']
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_data_df['logN'] = np.log(input_data_df['N'])
    input_data_df['logD'] = np.log(input_data_df['D'])

    with Pool(processes=cpu_count()) as pool:
        results = []
        for model in tqdm.tqdm(bootstrap_models):
            results.append(pool.apply_async(
                partial(predict_with_model, X_lr_key=X_lr_key, X_bs_key=X_bs_key),
                (input_data_df, model)))

        output_data_df_list = [result.get() for result in results]

    output_data_df = pd.concat(output_data_df_list, ignore_index=True)
    output_data_df['lr'] = np.exp(output_data_df['loglr'])
    output_data_df['bs'] = np.exp(output_data_df['logbs'])
    # also use mean model to predict
    mean_model_output_data_df = input_data_df.copy()
    lr_model_mean, bs_model_mean = mean_models
    mean_model_output_data_df['loglr'] = lr_model_mean.predict(input_data_df[X_lr_key])
    mean_model_output_data_df['logbs'] = bs_model_mean.predict(input_data_df[X_bs_key])
    mean_model_output_data_df['lr'] = np.exp(mean_model_output_data_df['loglr'])
    mean_model_output_data_df['bs'] = np.exp(mean_model_output_data_df['logbs'])

    return output_data_df, mean_model_output_data_df

def plot_lr_bs_pred(
    ax:plt.Axes, output_data_df:pd.DataFrame, mean_model_output_data_df:pd.DataFrame,
    mean_models:Tuple[LinearRegression, LinearRegression], random_seed:int=42
):
    random.seed(random_seed)
    sns.scatterplot(data=output_data_df, x='lr', y='bs', ax=ax, alpha=0.3)
    # plot mean_model_output_data_df scatter, and annotate the points with text
    sns.scatterplot(data=mean_model_output_data_df, x='lr', y='bs', ax=ax, color='red', s=100)
    for i, row in mean_model_output_data_df.iterrows():
        ax.text(
            row['lr']*1.1, row['bs']*random.uniform(0.85, 1.15),
            f"N={row['N']:.1e}, D={row['D']:.1e}\nlr={row['lr']:.2e}, bs={row['bs']:.2e}", fontsize=10, color='red'
        )
    # also plot the lr_model_mean and bs_model_mean text
    lr_model_mean, bs_model_mean = mean_models
    ax.text(transform=ax.transAxes, x=0.05, y=0.9, s=f"lr = {np.exp(lr_model_mean.intercept_):.3e}* N^{lr_model_mean.coef_[0]:.3e}* D^{lr_model_mean.coef_[1]:.3e}", fontsize=10)
    ax.text(transform=ax.transAxes, x=0.05, y=0.85, s=f"bs = {np.exp(bs_model_mean.intercept_):.3e}* N^{bs_model_mean.coef_[0]:.3e}* D^{bs_model_mean.coef_[1]:.3e}", fontsize=10)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.invert_yaxis()
    ax.grid(which='both')
    return ax

def fit_and_eval_lr_bs_scaling_model(data:pd.DataFrame, ax:plt.Axes, input_data_df:pd.DataFrame, seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        query_data = query_optimal_lr_bs_data(data)
        # do bootstrap fitting
        bootstrap_models = get_bootstrap_lr_bs_scaling_models(query_data)
        parameter_df = get_lr_bs_scaling_model_parameters(bootstrap_models)
        # calculate std and mean for each parameter
        lr_model_mean, bs_model_mean = get_mean_lr_bs_scaling_model(parameter_df)
        # print out the mean model
        print("Fitted Mean model:")
        print(f"log(lr) = {lr_model_mean.intercept_:.3e} + {lr_model_mean.coef_[0]:.3e} * log(N) + {lr_model_mean.coef_[1]:.3e} * log(D)")
        print(f"log(bs) = {bs_model_mean.intercept_:.3e} + {bs_model_mean.coef_[0]:.3e} * log(N) + {bs_model_mean.coef_[1]:.3e} * log(D)")
        output_data_df, mean_model_output_data_df = get_lr_bs_pred(input_data_df, bootstrap_models, (lr_model_mean, bs_model_mean))

        plot_lr_bs_pred(
            ax, output_data_df, mean_model_output_data_df, (lr_model_mean, bs_model_mean)
        )

    return bootstrap_models, (lr_model_mean, bs_model_mean)

@app.command()
def place_holder():
    print("place holder")

@app.command()
def pred_opt_lr_bs(
    model_params:float, data_in_token:float, seq_len:int, 
    parameter_version:str="1004"
):
    # TODO: in future, add more logic for output_data_df print
    exp_df = pd.DataFrame([
        {"N":model_params,"D":data_in_token, "seq_len":seq_len},
    ])
    parameter_version_2_path = {
        "1004":"1004_fitted_lr_bs_scaling_model_parameters.csv"
    }
    if parameter_version not in parameter_version_2_path:
        raise ValueError(f"parameter_version should be one of {parameter_version_2_path.keys()}")
    parameter_verion_2_X_keys = {
        "1004":[['logN','logD'],['logD']],
    }
    X_keys = parameter_verion_2_X_keys[parameter_version]

    parameter_df_path = Path(__file__).parent.parent / "data" /  parameter_version_2_path[parameter_version]
    parameter_df = pd.read_csv(parameter_df_path)
    bootstrap_models = load_bootstrap_lr_bs_models(
        parameter_df, X_lr_key=X_keys[0], X_bs_key=X_keys[1]
    )
    mean_models = get_mean_lr_bs_scaling_model(
        parameter_df, X_lr_key=X_keys[0], X_bs_key=X_keys[1]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output_data_df, mean_model_output_data_df = get_lr_bs_pred(
            exp_df,  bootstrap_models, mean_models, X_lr_key=X_keys[0], X_bs_key=X_keys[1]
        )

    exp_df['lr'] = mean_model_output_data_df['lr'].values
    exp_df['bs'] = mean_model_output_data_df['bs'].values
    exp_df['lr'] = exp_df['lr'].apply(lambda x: f"{x:.3e}")
    exp_df['bs_in_sample'] = exp_df.eval("bs/seq_len")
    # only keep lr, bs, bs_in_sample
    show_df = exp_df[['N','D','seq_len','lr', 'bs', 'bs_in_sample']]
    print(show_df.to_csv(sep=" ", index=False))
    return output_data_df, mean_model_output_data_df


if __name__ == "__main__":
    app()
