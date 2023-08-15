import plotly.graph_objects as go
#import seaborn as sns
import numpy as np


def plot_by_param(norms,model_names,accuracy_data,_colors,param_,filter_by_layers=False, n_layer=1):
    """
    Generate and display accuracy plots using plotly

    Parameters:
        norms (list): List of normalization methods.
        model_names (list): List of model names.
        accuracy_data (dict): Dictionary containing accuracy data for different models and norms.
        _colors (dict): Dictionary of colors for different elements.
        param_ (str): Parameter to plot against (e.g., "pooling", "nlayers", etc.).
        filter_by_layers (bool): If True, filter models by layers.
        n_layer (int): Layer number to filter models (if filter_by_layers is True).
    """
    #unique_param_values = list(set(param_values))
    #param_colors = sns.color_palette("husl", n_colors=len(unique_param_values)).as_hex()
    #param_color_map = dict(zip(unique_param_values, param_colors))
    for norm in norms:
        fig = go.Figure()
        for model_name in model_names:
            accuracies = accuracy_data[norm][model_name]
            eps, accs = zip(*accuracies)
            # Set the line style based on the model name
            line_dash = 'solid' if 'FC' not in model_name else 'dash'
            # Assign the same color shade based on the pooling option
            if param_ == "pooling":
                line_color = 'rgb(0, 0, 0)' if  'FC' in model_name else _colors[model_name.split('_')[3] == 'True']
                if filter_by_layers:
                    if int(model_name.split('_')[2]) == n_layer or 'FC' in model_name:
                        fig.add_trace(go.Scatter(x=eps, y=accs, mode='lines+markers', name=model_name, line=dict(dash=line_dash, color=line_color)))
                    else:
                        continue
            elif param_ == "nlayers":
                # Assign the same color shade based on the n_layers option
                n_layers = int(model_name.split('_')[2])
                line_color = 'rgb(0, 0, 0)' if 'FC' in model_name else _colors[n_layers]



            elif param_ == "kernels":
                line_color = 'rgb(42, 170, 138)' if 'FC' in model_name else _colors[int(model_name.split('_')[4])]
                if filter_by_layers:
                    if int(model_name.split('_')[2]) == n_layer or 'FC' in model_name:
                        fig.add_trace(go.Scatter(x=eps, y=accs, mode='lines+markers', name=model_name, line=dict(dash=line_dash, color=line_color)))
                    else:
                        continue
            elif param_ == "dense":
                line_color = 'rgb(0, 0, 0)' if  'FC' in model_name else _colors[model_name.split('_')[5] == 'True']
                if filter_by_layers:
                    if int(model_name.split('_')[2]) == n_layer or 'FC' in model_name:
                        fig.add_trace(go.Scatter(x=eps, y=accs, mode='lines+markers', name=model_name, line=dict(dash=line_dash, color=line_color)))
                    else:
                        continue
            if not filter_by_layers:
                fig.add_trace(go.Scatter(x=eps, y=accs, mode='lines+markers', name=model_name, line=dict(dash=line_dash, color=line_color)))
        if filter_by_layers:
            fig.update_layout(
                xaxis_title='Epsilon',
                yaxis_title='Accuracy',
                title=f'Accuracy on Attack Data (Norm={norm}, n_layers={n_layer})',
                showlegend=True
            )
        else:
            fig.update_layout(
                xaxis_title='Epsilon',
                yaxis_title='Accuracy',
                title=f'Accuracy on Attack Data (Norm={norm})',
                showlegend=True
            )
        fig.show()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_by_param_sns(norms, model_names, accuracy_data, _colors, param_, filter_by_layers=False, n_layer=1):
    """
    Generate and display accuracy plots using seaborn

    Parameters:
        norms (list): List of normalization methods.
        model_names (list): List of model names.
        accuracy_data (dict): Dictionary containing accuracy data for different models and norms.
        _colors (dict): Dictionary of colors for different elements.
        param_ (str): Parameter to plot against (e.g., "pooling", "nlayers", etc.).
        filter_by_layers (bool): If True, filter models by layers.
        n_layer (int): Layer number to filter models (if filter_by_layers is True).
    """
    for norm in norms:
        plt.figure(figsize=(10, 6))
        for model_name in model_names:
            accuracies = accuracy_data[norm][model_name]
            eps, accs = zip(*accuracies)

            line_style = '--' if 'FC' in model_name else '-'

            if param_ == "pooling":
                line_color = 'black' if 'FC' in model_name else _colors[model_name.split('_')[3] == 'True']
                if filter_by_layers:
                    if int(model_name.split('_')[2]) == n_layer or 'FC' in model_name:
                        plt.plot(eps, accs, label=model_name, linestyle=line_style, color=line_color)
                    else:
                        continue
            elif param_ == "nlayers":
                n_layers = int(model_name.split('_')[2])
                line_color = 'black' if 'FC' in model_name else _colors[n_layers]
            elif param_ == "kernels":
                line_color = (0.165, 0.631, 0.514) if 'FC' in model_name else _colors[int(model_name.split('_')[4])]
                if filter_by_layers:
                    if int(model_name.split('_')[2]) == n_layer or 'FC' in model_name:
                        plt.plot(eps, accs, label=model_name, linestyle=line_style, color=line_color)
                    else:
                        continue
            elif param_ == "dense":
                line_color = 'black' if 'FC' in model_name else _colors[model_name.split('_')[5] == 'True']
                if filter_by_layers:
                    if int(model_name.split('_')[2]) == n_layer or 'FC' in model_name:
                        plt.plot(eps, accs, label=model_name, linestyle=line_style, color=line_color)
                    else:
                        continue
            if not filter_by_layers:
                plt.plot(eps, accs, label=model_name, linestyle=line_style, color=line_color)

        plt.xlabel('Epsilon')
        plt.ylabel('Accuracy')
        if filter_by_layers:
            plt.title(f'Accuracy on Attack Data (Norm={norm}, n_layers={n_layer})')
        else:
            plt.title(f'Accuracy on Attack Data (Norm={norm})')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()


def plot_boxplots(data, x_labels, x_label, y_label, title):
    """
    Generate and display boxplots based on provided data.

    Parameters:
        data (list of lists): List containing data for boxplots.
        x_labels (list): Labels for x-axis ticks.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title for the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data)

    plt.xticks(range(1, len(x_labels) + 1), x_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def plot_violinplots(data, x_labels, x_label, y_label, title):
    """
    Generate and display violin plots based on provided data.

    Parameters:
        data (list of lists): List containing data for violin plots.
        x_labels (list): Labels for x-axis ticks.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title for the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 7))
    vp = ax.violinplot(data, showmedians=True)

    plt.xticks(range(1, len(x_labels) + 1), x_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def get_acc_consi_epsi(model_consistency_dict,accuracy_data, norm = np.inf, epsilon_ = 0.1):
    acc_consistency=[]
    accuracy_data_ = {model: values for model, values in accuracy_data[np.inf].items() if model != "simple_FC_256"}
    acc_epsilon = {model: value for model, values in accuracy_data_.items() for epsilon, value in values if epsilon == epsilon_}
    for model_name, model_consistency in model_consistency_dict.items():
        if model_name in acc_epsilon:
            acc_consistency.append((model_name,acc_epsilon[model_name],model_consistency))
    acc_consistency.append(('simple_FC_256',[acc for epsilon, acc in accuracy_data[np.inf]['simple_FC_256'] if epsilon == epsilon_][0],model_consistency_dict['simple_FC_256']))
    return acc_consistency

import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
def plot_acc_consi(acc_consistency, conf_interval = False, mean_lines = False, fc_lines = True):


    # Your data
    x = np.array([consistency for _, _, consistency in acc_consistency])
    y = np.array([accuracy for _, accuracy, _ in acc_consistency])
    labels = [model_name for model_name, _, _ in acc_consistency]

    # Fit a linear regression line
    regression_model = LinearRegression()
    regression_model.fit(x.reshape(-1, 1), y)
    regression_line = regression_model.predict(x.reshape(-1, 1))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Calculate the confidence interval
    if conf_interval:
        confidence_interval = stats.t.ppf(0.975, len(x) - 2) * std_err

    plt.figure(figsize=(20, 15))
    scatter = plt.scatter(x[:-1], y[:-1], color='blue')
    scatter = plt.scatter(x[-1], y[-1], color='green')
    plt.plot(x, regression_line, linestyle='dashed', color='red', label='Regression Line')
    if conf_interval:
        # Plot the shaded confidence interval
        plt.fill_between(x, regression_line - confidence_interval, regression_line + confidence_interval, color='red', alpha=0.2, label='Confidence Interval')

    annotations = [plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center') for i, label in enumerate(labels)]

    # Adjust the positions of the annotations to avoid overlapping
    adjust_text(annotations)

    if mean_lines:
        # Add dashed horizontal and vertical lines
        plt.axhline(y=np.mean(y), color='green', linestyle='dashed', label='Mean Accuracy')
        plt.axvline(x=np.mean(x), color='blue', linestyle='dashed', label='Mean Consistency')
    if fc_lines:
        # Add dashed horizontal and vertical lines
        plt.axhline(y=y[-1], color='cyan', linestyle='dashed', label='FC Accuracy')
        plt.axvline(x=x[-1], color='cyan', linestyle='dashed', label='FC Consistency')

    plt.xlabel('Consistency')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Consistency')

    # Add regression coefficients and standard error to the plot
    plt.text(0.95, 0.8, f'Regression Coefficients: {regression_model.coef_[0]:.4f}', transform=plt.gca().transAxes, horizontalalignment='right')
    plt.text(0.95, 0.75, f'Standard Error: {std_err:.4f}', transform=plt.gca().transAxes, horizontalalignment='right')

    plt.legend()
    plt.show()


def get_acc_consi_epsi_fashion(model_consistency_dict,accuracy_data, norm = np.inf, epsilon_ = 0.1):
    acc_consistency=[]
    accuracy_data_ = {model: values for model, values in accuracy_data[np.inf].items() if model not in ["simple_FC_2_256","simple_FC_3_512" ]}
    acc_epsilon = {model: value for model, values in accuracy_data_.items() for epsilon, value in values if epsilon == epsilon_}
    for model_name, model_consistency in model_consistency_dict.items():
        if model_name in acc_epsilon:
            acc_consistency.append((model_name,acc_epsilon[model_name],model_consistency))
    acc_consistency.append(('simple_FC_2_256',[acc for epsilon, acc in accuracy_data[np.inf]['simple_FC_2_256'] if epsilon == epsilon_][0],model_consistency_dict['simple_FC_2_256']))
    acc_consistency.append(('simple_FC_3_512',[acc for epsilon, acc in accuracy_data[np.inf]['simple_FC_3_512'] if epsilon == epsilon_][0],model_consistency_dict['simple_FC_3_512']))
    return acc_consistency


def plot_acc_consi_fashion(acc_consistency, conf_interval = False, mean_lines = False, fc_lines = True):


    # Your data
    x = np.array([consistency for _, _, consistency in acc_consistency])
    y = np.array([accuracy for _, accuracy, _ in acc_consistency])
    labels = [model_name for model_name, _, _ in acc_consistency]

    # Fit a linear regression line
    regression_model = LinearRegression()
    regression_model.fit(x.reshape(-1, 1), y)
    regression_line = regression_model.predict(x.reshape(-1, 1))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Calculate the confidence interval
    if conf_interval:
        confidence_interval = stats.t.ppf(0.975, len(x) - 2) * std_err

    plt.figure(figsize=(20, 15))
    scatter = plt.scatter(x[:-2], y[:-2], color='blue')
    scatter = plt.scatter(x[-1], y[-1], color='green')
    scatter = plt.scatter(x[-2], y[-2], color='green')
    plt.plot(x, regression_line, linestyle='dashed', color='red', label='Regression Line')
    if conf_interval:
        # Plot the shaded confidence interval
        plt.fill_between(x, regression_line - confidence_interval, regression_line + confidence_interval, color='red', alpha=0.2, label='Confidence Interval')

    annotations = [plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center') for i, label in enumerate(labels)]

    # Adjust the positions of the annotations to avoid overlapping
    adjust_text(annotations)

    if mean_lines:
        # Add dashed horizontal and vertical lines
        plt.axhline(y=np.mean(y), color='green', linestyle='dashed', label='Mean Accuracy')
        plt.axvline(x=np.mean(x), color='blue', linestyle='dashed', label='Mean Consistency')
    if fc_lines:
        # Add dashed horizontal and vertical lines
        plt.axhline(y=y[-1], color='cyan', linestyle='dashed', label='FC1 Accuracy')
        plt.axvline(x=x[-1], color='cyan', linestyle='dashed', label='FC1 Consistency')
        plt.axhline(y=y[-2], color='cyan', linestyle='dashed', label='FC2 Accuracy')
        plt.axvline(x=x[-2], color='cyan', linestyle='dashed', label='FC2 Consistency')

    plt.xlabel('Consistency')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Consistency')

    # Add regression coefficients and standard error to the plot
    plt.text(0.95, 0.8, f'Regression Coefficients: {regression_model.coef_[0]:.4f}', transform=plt.gca().transAxes, horizontalalignment='right')
    plt.text(0.95, 0.75, f'Standard Error: {std_err:.4f}', transform=plt.gca().transAxes, horizontalalignment='right')

    plt.legend()
    plt.show()


def get_acc_consi_epsi_best(model_consistency_dict,accuracy_data, norm = np.inf, epsilon_ = 0.1):
    acc_consistency=[]

    acc_epsilon = {model: value for model, values in accuracy_data[np.inf].items() for epsilon, value in values if epsilon == epsilon_}
    for model_name, model_consistency in model_consistency_dict.items():
        if model_name in acc_epsilon:
            acc_consistency.append((model_name,acc_epsilon[model_name],model_consistency))

    return acc_consistency

def plot_acc_consi_best(acc_consistency, conf_interval = False, mean_lines = False, fc_lines = True, reg = True,annote=True,limit= False):


    # Your data
    x = np.array([consistency for _, _, consistency in acc_consistency])
    y = np.array([accuracy for _, accuracy, _ in acc_consistency])
    labels = [model_name for model_name, _, _ in acc_consistency]

    # Fit a linear regression line
    # Here we should remove the adversarial model from the regression
    regression_model = LinearRegression()
    regression_model.fit(x[:-1].reshape(-1, 1), y[:-1])
    regression_line = regression_model.predict(x[:-1].reshape(-1, 1))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:-1], y[:-1])
    # Calculate the confidence interval
    if conf_interval:
        confidence_interval = stats.t.ppf(0.975, len(x[:-1]) - 2) * std_err

    plt.figure(figsize=(20, 15))
    scatter = plt.scatter(x[2:], y[2:], color='blue')
    scatter = plt.scatter(x[0], y[0], color='green')
    scatter = plt.scatter(x[1], y[1], color='green')
    if reg :
        plt.plot(x[:-1], regression_line, linestyle='dashed', color='red', label='Regression Line')
    if conf_interval:
        # Plot the shaded confidence interval
        plt.fill_between(x[:-1], regression_line - confidence_interval, regression_line + confidence_interval, color='red', alpha=0.2, label='Confidence Interval')
    if annote:
        annotations = [plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center') for i, label in enumerate(labels)]

        # Adjust the positions of the annotations to avoid overlapping
        adjust_text(annotations)

    if mean_lines:
        # Add dashed horizontal and vertical lines
        plt.axhline(y=np.mean(y), color='green', linestyle='dashed', label='Mean Accuracy')
        plt.axvline(x=np.mean(x), color='blue', linestyle='dashed', label='Mean Consistency')
    if fc_lines:
        # Add dashed horizontal and vertical lines
        plt.axhline(y=y[0], color='cyan', linestyle='dashed', label='FC1 Accuracy')
        plt.axvline(x=x[0], color='cyan', linestyle='dashed', label='FC1 Consistency')
        plt.axhline(y=y[1], color='cyan', linestyle='dashed', label='FC2 Accuracy')
        plt.axvline(x=x[1], color='cyan', linestyle='dashed', label='FC2 Consistency')
    if limit:
        plt.xlim([0, 1])
        plt.ylim([0, 1])

    plt.xlabel('Consistency')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Consistency')

    # Add regression coefficients and standard error to the plot
    plt.text(0.95, 0.8, f'Regression Coefficients: {regression_model.coef_[0]:.4f}', transform=plt.gca().transAxes, horizontalalignment='right')
    plt.text(0.95, 0.75, f'Standard Error: {std_err:.4f}', transform=plt.gca().transAxes, horizontalalignment='right')

    plt.legend()
    plt.show()