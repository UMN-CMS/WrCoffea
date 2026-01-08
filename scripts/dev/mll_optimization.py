import ROOT
import matplotlib.pyplot as plt
import os
import numpy as np
import mplhep as hep
import array
from collections import OrderedDict
#import histogram_configs
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
import logging

hep.style.use("CMS")

# Constants for configuration
BKG_FILE_PATH = "root_outputs/hists/mll_optimization_0815/all_bkgs.root"
SIG_FILE_PATH = "root_outputs/hists/mll_optimization_0815/all_sigs_newest.root"
OUTPUT_DIR = "plotting/mll_optimization_study"
TARGET_HISTOGRAM_NAME = "mass_dileptons_fourobject"
TARGET_MLL = "150mll"
TARGET_CHANNEL = "eejj"
SCALING_FACTOR = 59.74 * 1000
BASE_CONSTANT = 3 / 2

# Configure logging
logging.basicConfig(level=logging.INFO)

def ensure_directory_exists(path):
    """Ensure that the output directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def open_root_file(file_path):
    """Open a ROOT file and handle potential errors."""
    try:
        root_file = ROOT.TFile(file_path)
        if not root_file or root_file.IsZombie():
            raise FileNotFoundError(f"Failed to open ROOT file: {file_path}")
        logging.info(f"Opened ROOT file: {file_path}")
        return root_file
    except Exception as e:
        logging.error(f"Error opening ROOT file: {e}")
        raise

# Function to recursively get histograms from a directory in a ROOT file
def get_histograms(directory, path=""):
    histograms = {}
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        obj_path = os.path.join(path, obj.GetName())
        if obj.IsA().InheritsFrom("TDirectory"):
            histograms.update(get_histograms(obj, obj_path))
        elif obj.IsA().InheritsFrom("TH1"):
            histograms[obj_path] = obj
    return histograms

def organize_histograms(histograms):
    organized_histograms = {}
    for path, hist in histograms.items():
        components = path.split(os.sep)

        process = components[0]
        channel = components[1]
        mll = components[2]
        hist_name = components[3]

        key = (channel, mll, hist_name)

        if key not in organized_histograms:
            organized_histograms[key] = {}
        organized_histograms[key][process] = hist

    return organized_histograms

def process_histograms(organized_bkg_histograms, organized_signal_histograms):
    """Process histograms based on target criteria."""
    for key, bkg_hist_dict in organized_bkg_histograms.items():
        channel, mll, hist_name = key

        # Only proceed if the histogram matches the target criteria
        if hist_name == TARGET_HISTOGRAM_NAME and mll == TARGET_MLL and channel == TARGET_CHANNEL:
            grouped_bkg_histograms = group_histograms(bkg_hist_dict)
            reordered_bkg_histograms = reorder_dict(mll, grouped_bkg_histograms)

            # Process signal histograms that match the criteria
            process_signal_histograms(channel, mll, hist_name, reordered_bkg_histograms, organized_signal_histograms)

def group_histograms(hist_dict):
    stacked_histograms = {}
        
    groupings = {
        'Z+jets': ['DYJets'],
        'tt+tW': ['tt+tW'],
        'Nonprompt': ['tt_semileptonic', 'WJets', 'SingleTop'],
        'Other Backgrounds': ['ttX', 'Diboson', 'Triboson']
    }

    for group_name, processes in groupings.items():
        stacked_histogram = None
        for process in processes:
            if process in hist_dict:
                hist = hist_dict[process]
                if stacked_histogram is None:
                    stacked_histogram = hist.Clone()
                else:
                    stacked_histogram.Add(hist)
        if stacked_histogram is not None:
            stacked_histograms[group_name] = stacked_histogram
    return stacked_histograms

def reorder_dict(mll: str, original_dict: dict) -> dict:
    desired_order = {
        "60mll150": ['Other Backgrounds', 'Nonprompt', 'tt+tW', 'Z+jets'],
        "150mll400": ['Other Backgrounds', 'Nonprompt', 'tt+tW', 'Z+jets'],
        "150mll": ['Other Backgrounds', 'Nonprompt', 'tt+tW', 'Z+jets'],
        "400mll": ['Other Backgrounds', 'Nonprompt', 'Z+jets', 'tt+tW'],
    }

    order = desired_order.get(mll)
    if order is None:
        return original_dict

    reordered_dict = OrderedDict((key, original_dict[key]) for key in order if key in original_dict)
    return reordered_dict

def process_signal_histograms(channel, mll, hist_name, reordered_bkg_histograms, organized_signal_histograms):
    """Loop over signal histograms and calculate efficiencies."""
    for signal_key, signal_hist_dict in organized_signal_histograms.items():
        sig_channel, sig_mll, sig_hist_name = signal_key

        # Match signal histograms based on target criteria
        if sig_hist_name == TARGET_HISTOGRAM_NAME and sig_mll == mll and sig_channel == channel:
            event_weights_key = (sig_channel, sig_mll, "event_weight")
            sig_hists_event_dict = organized_signal_histograms.get(event_weights_key, {})

            # Calculate efficiencies for different mll thresholds
            sig_mean, sig_stddev, fit_mean, fit_stddev, efficiencies, efficiency_errors, thresholds = loop_over_mll_thresholds(
                channel, mll, hist_name, reordered_bkg_histograms, signal_hist_dict, sig_hists_event_dict
            )
            plot_efficiency_vs_thresholds(channel, mll, hist_name, thresholds, efficiencies, efficiency_errors)

def loop_over_mll_thresholds(channel, mll, hist_name, bkg_hist_dict, sig_hist_dict, sig_hists_event_dict, min_mll=150, max_mll=1050, step=100):
    """Loop over mll thresholds, calculate efficiencies, and store results."""
    thresholds = np.arange(min_mll, max_mll + step, step)

    # Initialize result containers
    results = {
        'sig_mean': [],
        'sig_stddev': [],
        'fit_mean': [],
        'fit_stddev': [],
        'sig_efficiencies': [],
        'sig_efficiencies_err': [],
        'thresholds': thresholds
    }

    # Combine background histograms
    combined_bkg_hist = combine_histograms(bkg_hist_dict)

    # Process signal histograms
    for sig_name, sig_hist in sig_hist_dict.items():
        if sig_name == "MWR2000_MN1000":  # Optional: Make this a parameter
            event_weights_hist = sig_hists_event_dict.get(sig_name)

            signal_results = process_signal_histogram(
                channel, mll, hist_name, combined_bkg_hist, sig_hist, event_weights_hist, thresholds, sig_name
            )

            # Store results for this signal
            results['sig_mean'].append(signal_results['sig_mean'])
            results['sig_stddev'].append(signal_results['sig_stddev'])
            results['fit_mean'].append(signal_results['fit_mean'])
            results['fit_stddev'].append(signal_results['fit_stddev'])
            results['sig_efficiencies'].append(signal_results['sig_efficiencies'])
            results['sig_efficiencies_err'].append(signal_results['sig_efficiencies_err'])

    return results['sig_mean'], results['sig_stddev'], results['fit_mean'], results['fit_stddev'], results['sig_efficiencies'], results['sig_efficiencies_err'], thresholds


def combine_histograms(bkg_hist_dict):
    """Combine all background histograms into a single histogram."""
    combined_hist = None
    for process, hist in bkg_hist_dict.items():
        if combined_hist is None:
            combined_hist = hist.Clone("combined_bkg_hist")
        else:
            combined_hist.Add(hist)
    return combined_hist


def process_signal_histogram(channel, mll, hist_name, combined_bkg_hist, sig_hist, event_weights_hist, thresholds, sig_sample):
    """Process a signal histogram across different mll thresholds."""
    sig_mean, sig_stddev, fit_mean, fit_stddev = [], [], [], []
    sig_efficiencies, sig_efficiencies_err = [], []
    bkg_err = []
    # Loop over mll thresholds
    for mll_threshold in thresholds[:-1]:
        logging.info(f"\nmll > {mll_threshold}: {channel}")

        # Fit Gaussian for this threshold
        fit_result = fit_gaussian_for_mll_threshold(channel, mll, hist_name, sig_hist, event_weights_hist, mll_threshold)

        # Access values from the dictionary
        signal_mean = fit_result['mean']
        signal_stddev = fit_result['stddev']
        fitted_mean = fit_result['fitted_mean']
        fitted_stddev = fit_result['fitted_stddev']
        bin_edges = fit_result['bin_centers']
        y_vals = fit_result['y_vals']
        lower_bound = fit_result['lower_bound']
        upper_bound = fit_result['upper_bound']

        bkg_sigma = plot_background_mlljj(channel, mll, hist_name, combined_bkg_hist, mll_threshold, lower_bound, upper_bound, fitted_mean, fitted_stddev)

        # Calculate efficiency
        efficiency_result  = get_efficiency_for_mll_threshold(
            channel, mll, hist_name, combined_bkg_hist, sig_hist, event_weights_hist, mll_threshold, lower_bound, upper_bound, bkg_sigma
        )

        # Access the efficiency and error from the dictionary
        efficiency = efficiency_result['efficiency']
        error = efficiency_result['efficiency_error']

        # Optionally, plot histogram if threshold is 150
        plot_signal_peak(channel, mll_threshold, hist_name, bin_edges, y_vals, signal_mean, signal_stddev, sig_sample)

        # Append results for this threshold
        sig_mean.append(signal_mean)
        sig_stddev.append(signal_stddev)
        fit_mean.append(fitted_mean)
        fit_stddev.append(fitted_stddev)
        sig_efficiencies.append(efficiency)
        sig_efficiencies_err.append(error)
        bkg_err.append(bkg_sigma)

    return {
        'sig_mean': sig_mean,
        'sig_stddev': sig_stddev,
        'fit_mean': fit_mean,
        'fit_stddev': fit_stddev,
        'sig_efficiencies': sig_efficiencies,
        'sig_efficiencies_err': sig_efficiencies_err
    }

def fit_gaussian_for_mll_threshold(channel, mll, hist_name, sig_hist, event_weights_hist, threshold, initial_guess_stddev=10):
    """Fit a Gaussian to the signal histogram filtered by the mll threshold."""
    
    # Filter the histogram by mll threshold
    sig_hist_filtered = filter_hist_by_mll(sig_hist, threshold)

    # Compute statistics (mean and stddev) from the filtered histogram
    mean, stddev, bin_centers, y_vals, _ = compute_stats(sig_hist_filtered)
    
    # Initial guess for the fit parameters [amplitude, mean, stddev]
    initial_guess = [max(y_vals), bin_centers[np.argmax(y_vals)], initial_guess_stddev]
    
    try:
        # Fit the Gaussian function
        popt, pcov = curve_fit(gaussian, bin_centers, y_vals, p0=initial_guess)
        amplitude, mu, fitted_stddev = popt
        
        logging.info(f"Gaussian params: Amplitude = {amplitude}, Mean = {mu}, Stddev = {fitted_stddev}")
        
        # Calculate the lower and upper bounds based on the fit
        lower_bound = mu - 2 * fitted_stddev
        upper_bound = mu + 2 * fitted_stddev
        logging.info(f"(Lower, Upper) bounds = ({round(lower_bound, 2)}, {round(upper_bound, 2)})")
        
    except RuntimeError as e:
        # Handle cases where the curve fitting fails
        logging.error(f"Curve fitting failed for {hist_name} at threshold {threshold}: {e}")
        # Return NaN values or fallback results in case of a fit failure
        amplitude, mu, fitted_stddev = np.nan, np.nan, np.nan
        lower_bound, upper_bound = np.nan, np.nan
    
    # Return results in a dictionary for easy access
    return {
        'mean': mean,
        'stddev': stddev,
        'fitted_mean': mu,
        'fitted_stddev': fitted_stddev,
        'bin_centers': bin_centers,
        'y_vals': y_vals,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def filter_hist_by_mll(hist, mll_threshold):
    """Filter the histogram by mll threshold and scale content."""
    
    # Clone the histogram and reset it for modification
    filtered_hist = hist.Clone("filtered_hist")
    filtered_hist.Reset()  # Clear the contents of the cloned histogram

    n_bins_mll = hist.GetNbinsX()
    n_bins_mlljj = hist.GetNbinsY()

    # Loop through mll bins
    for mll_bin in range(1, n_bins_mll + 1):
        mll_center = hist.GetXaxis().GetBinCenter(mll_bin)

        # Stop if mll_center exceeds the threshold for efficiency
        if mll_center <= mll_threshold:
            continue

        # Process only the bins greater than the threshold
        for mlljj_bin in range(1, n_bins_mlljj + 1):
            content = hist.GetBinContent(mll_bin, mlljj_bin)
            error = hist.GetBinError(mll_bin, mlljj_bin)

            # Set scaled content and error in the filtered histogram
            filtered_hist.SetBinContent(mll_bin, mlljj_bin, content * SCALING_FACTOR)
            filtered_hist.SetBinError(mll_bin, mlljj_bin, error * SCALING_FACTOR)

    return filtered_hist

def compute_stats(hist):
    """Compute the weighted mean and standard deviation for the y-axis (mlljj) of a 2D histogram."""

    n_bins_mll = hist.GetNbinsX()
    n_bins_mlljj = hist.GetNbinsY()

    # Initialize arrays to hold the centers and weights for the mlljj axis
    mlljj_weights = np.zeros(n_bins_mlljj)
    mlljj_centers = np.zeros(n_bins_mlljj)
    mlljj_low_bin_edge = np.zeros(n_bins_mlljj)

    # Loop over the mlljj bins and sum the content over all mll bins
    for mlljj_bin in range(1, n_bins_mlljj + 1):
        y_bin_content = 0
        for mll_bin in range(1, n_bins_mll + 1):
            content = hist.GetBinContent(mll_bin, mlljj_bin)
            y_bin_content += content

        # Store the total weight and bin center
        mlljj_weights[mlljj_bin - 1] = y_bin_content
        mlljj_centers[mlljj_bin - 1] = hist.GetYaxis().GetBinCenter(mlljj_bin)
        mlljj_low_bin_edge[mlljj_bin - 1] = hist.GetYaxis().GetBinLowEdge(mlljj_bin)

    # Compute the weighted mean and standard deviation using NumPy
    mean = np.average(mlljj_centers, weights=mlljj_weights)
    variance = np.average((mlljj_centers - mean) ** 2, weights=mlljj_weights)
    stddev = np.sqrt(variance)

    return mean, stddev, mlljj_centers, mlljj_weights, mlljj_low_bin_edge

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def get_efficiency_for_mll_threshold(channel, mll, hist_name, combined_hist, sig_hist, event_weights_hist, threshold, lower_bound, upper_bound, bkg_err):
    """Calculate and return the efficiency histogram and plot it as a step plot."""

    # Filter the histograms by the mll threshold
    sig_hist = filter_hist_by_mll(sig_hist, threshold)
    bkg_hist = filter_hist_by_mll(combined_hist, threshold)

    # Integrate signal and background histograms within the bounds
    signal_integral, signal_error = integrate_histogram_within_bounds(sig_hist, lower_bound, upper_bound)
    background_integral, background_error = integrate_histogram_within_bounds(bkg_hist, lower_bound, upper_bound)

    logging.info(f"Background Integral: {background_integral}")
    logging.info(f"Signal Integral: {signal_integral}")

    # Integrate the event weights histogram over its entire range
    event_weights_integral = event_weights_hist.Integral(1, event_weights_hist.GetNbinsX()) * 59.74 * 1000

    logging.info(f"Signal Sum of Weights: {round(event_weights_integral, 2)}")

    # Adjust signal integral by the integral of event weights, avoid division by zero
    if event_weights_integral != 0:
        signal_integral, signal_error = adjust_by_event_weights(signal_integral, signal_error, event_weights_integral)
    else:
        logging.warning("Event weights integral is zero, cannot scale signal integral.")
        return {'efficiency': 0, 'efficiency_error': 0}

    logging.info(f"Signal Efficiency: {signal_integral:.2e}")

    # Calculate the efficiency and error on the efficiency
    efficiency, efficiency_error = error_on_ratio(signal_integral, signal_error, background_integral, background_error, bkg_err)

    return {
        'efficiency': efficiency,
        'efficiency_error': efficiency_error
    }

def integrate_histogram_within_bounds(hist, lower_bound, upper_bound):
    """Integrate the histogram content and its error between lower and upper bounds on the y-axis."""

    # Validate bounds
    if lower_bound > upper_bound:
        logging.warning(f"Invalid bounds: lower_bound ({lower_bound}) is greater than upper_bound ({upper_bound}).")
        return {'integral': 0.0, 'error': 0.0}

    n_bins_x = hist.GetNbinsX()
    n_bins_y = hist.GetNbinsY()

    integral = 0.0
    error_squared = 0.0

    # Loop over all bins in x and y dimensions
    for y_bin in range(1, n_bins_y + 1):
        y_center = hist.GetYaxis().GetBinCenter(y_bin)

        # Early exit if y_center exceeds upper_bound
        if y_center > upper_bound:
            break

        # Only process bins within the bounds
        if lower_bound <= y_center <= upper_bound:
            for x_bin in range(1, n_bins_x + 1):
                content = hist.GetBinContent(x_bin, y_bin)
                error = hist.GetBinError(x_bin, y_bin)

                # Accumulate integral and error
                integral += content
                error_squared += error ** 2

    total_error = np.sqrt(error_squared)

    # Return results in a dictionary
    return integral, total_error
    

def adjust_by_event_weights(signal_integral, signal_error, event_weights_integral):
    """Adjust the signal integral and error by the event weights integral."""
    signal_integral /= event_weights_integral
    signal_error /= event_weights_integral
    return signal_integral, signal_error

def error_on_ratio(integral1, error1, integral2, error2, bkg_err):
    """Calculate the ratio and the error on the ratio integral1 / sqrt(BASE_CONSTANT + integral2)."""

    # Handle case where integral2 is zero or negative to avoid invalid sqrt
    if integral2 < 0:
        logging.error(f"Invalid integral2: {integral2}. Must be non-negative.")
        return {'ratio': np.nan, 'error_ratio': np.nan}

    background_sqrt = np.sqrt(integral2)
#    denominator = BASE_CONSTANT + background_sqrt 
    denominator = BASE_CONSTANT + background_sqrt + bkg_err

    # Avoid division by zero in case the denominator is zero
    if denominator == 0:
        logging.error("Denominator is zero, cannot calculate ratio.")
        return {'ratio': np.nan, 'error_ratio': np.nan}

    ratio = integral1 / denominator

    # Partial derivatives for error propagation
    partial1 = 1 / denominator
    partial2 = -0.5 * integral1 / (background_sqrt * denominator**2)

    # Calculate the propagated error
    error_ratio = np.sqrt((partial1 * error1)**2 + (partial2 * error2)**2)

    logging.info(f"eff/(3/2 + sqrt(B) + bkg_err) = {ratio} +/- {error_ratio}")
#    logging.info(f"eff/(3/2 + sqrt(B)) = {ratio} +/- {error_ratio}")

    return ratio, error_ratio

def setup_plot(ax, xlabel, ylabel, xlim=None, ylim=None, legend_loc='best'):
    """Helper function to setup plot axis labels, limits, and legends."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis='y', useMathText=True)
    ax.legend(loc=legend_loc)

def rebin_2d_histogram(hist, rebin_factor_y):
    """Rebin the 2D ROOT histogram by the given rebin factor for the Y axis (mlljj)."""
    # Rebin only along the Y axis (mlljj axis) with the specified factor
    rebinned_hist = hist.Clone()
    rebinned_hist.RebinY(rebin_factor_y)
    return rebinned_hist

def exp_decreasing(x, a, b):
    return a * np.exp(-b * x)

def plot_background_mlljj(channel, mll, hist_name, combined_bkg_hist, mll_threshold, low_bound, high_bound, mean, stddev, output_dir=OUTPUT_DIR):

    # Filter the histograms by the mll threshold
    combined_bkg_hist = filter_hist_by_mll(combined_bkg_hist, mll_threshold)

    # Rebin the 2D ROOT histogram along the Y axis (mlljj axis)
    rebin_factor_y = 5  # Rebin Y axis by 10 to get 50 GeV bins (10 GeV * 5)
    rebinned_hist = rebin_2d_histogram(combined_bkg_hist, rebin_factor_y)

    mu, sigma, mlljj_centers, mlljj_weights, new_bins = compute_stats(rebinned_hist)

    fig, ax = plt.subplots()

    uncertainties = np.sqrt(mlljj_weights)

    # Plot the background data
    hep.histplot(
        mlljj_weights[:-1],
        bins=new_bins,
        yerr=uncertainties[:-1],
        histtype='step',
        color='green',
        label=f'All Backgrounds',
        linewidth=1,
        ax=ax
    )
    
    low_mask = (mlljj_centers >= mean - 4*stddev) & (mlljj_centers <= low_bound)
    high_mask = (mlljj_centers >= high_bound) & (mlljj_centers <= mean + 4*stddev)

    x_fit = np.concatenate([mlljj_centers[low_mask], mlljj_centers[high_mask]])
    y_fit = np.concatenate([mlljj_weights[low_mask], mlljj_weights[high_mask]])
    err = np.concatenate([uncertainties[low_mask], uncertainties[high_mask]])

    a_initial = np.max(y_fit)
    b_initial = 0.001
    # Fit the model using the first and last thirds of the data
    popt, pcov = curve_fit(exp_decreasing, x_fit, y_fit, sigma=err, p0=[a_initial, b_initial])

    y_fitted = exp_decreasing(mlljj_centers, *popt)

    # Define the range to plot based on the low and high masks
    fit_range_mask = (mlljj_centers >= mlljj_centers[low_mask][0]) & (mlljj_centers <= mlljj_centers[high_mask][-1])

    # Slice the fitted y values and corresponding x values (mlljj_centers)
    mlljj_centers_fit_range = mlljj_centers[fit_range_mask]
    y_fitted_range = y_fitted[fit_range_mask]

    # Plot the fitted curve only in the specified range
    ax.plot(mlljj_centers_fit_range, y_fitted_range, label=f'Fit', color='blue', lw=2)

    plt.axvspan(mlljj_centers[low_mask][-1], mlljj_centers[high_mask][0], color='gray', alpha=0.3, label=f"$2\sigma$ of Signal Peak (excluded in fit)")

    # Extract the parameter uncertainties (square root of the diagonal of the covariance matrix)
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    # Calculate the uncertainty in the fit at each x value using error propagation
    dy_da = np.exp(-b_fit * mlljj_centers_fit_range)       # ∂y/∂a = exp(-b * x)
    dy_db = -a_fit * mlljj_centers_fit_range * np.exp(-b_fit * mlljj_centers_fit_range)  # ∂y/∂b = -a * x * exp(-b * x)

    # Total uncertainty using error propagation formula
    y_uncertainty = np.sqrt((dy_da * a_err)**2 + (dy_db * b_err)**2)

    # Upper and lower bounds for the fit
    y_fitted_upper = y_fitted_range + y_uncertainty
    y_fitted_lower = y_fitted_range - y_uncertainty

    # Plot uncertainty band as a shaded region
    plt.fill_between(mlljj_centers_fit_range, y_fitted_lower, y_fitted_upper, color='blue', alpha=0.2)

    # Step 1: Define the excluded region (between the low and high masks)
    excluded_region_mask = (mlljj_centers >= mlljj_centers[low_mask][-1]) & (mlljj_centers <= mlljj_centers[high_mask][0])
    mlljj_centers_excluded = mlljj_centers[excluded_region_mask]
    mlljj_weights_excluded = mlljj_weights[excluded_region_mask]

    # Step 2: Get the uncertainty values in the excluded region
    y_uncertainty_excluded = y_uncertainty[(mlljj_centers_fit_range >= mlljj_centers[low_mask][-1]) & (mlljj_centers_fit_range <= mlljj_centers[high_mask][0])]
    background_int_excluded = np.sum(mlljj_weights_excluded)

    total_uncertainty_sum = np.sum(y_uncertainty_excluded)
    print(f"Total sum of the uncertainties in the excluded region: {total_uncertainty_sum}")
    print(f"Relative uncertianty: {total_uncertainty_sum/background_int_excluded}")

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e4)
#    ax.set_ylim(0, 1e3)
    ax.set_xlim(0,6000)

    # Add axis labels, title, and legends
    ax.set_xlabel(r"$m_{lljj} \mathrm{~[GeV]}$")
    ax.set_ylabel(r"Events / 50 GeV")
    hep.cms.label(loc=0, ax=ax)
    # Create custom legend with specific font size
    legend_fontsize = 16  # Set desired legend font size here
    legend = ax.legend(fontsize=legend_fontsize, loc='upper right')  # Apply font size to the legend

    # Add the text box in the top-left corner
    if channel == "eejj":
        flavor = r"$ee$"
        region = fr"$m_{{ll}} > {mll_threshold} \mathrm{{~GeV}}$"
    ax.text(0.05, 0.96, flavor, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')
    ax.text(0.05, 0.91, region, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')

    output_dir = f'{output_dir}/plots/mll{mll_threshold}'
    # Ensure the directory exists before saving the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = f'{output_dir}/mlljj_bkg_{channel}_mll{mll_threshold}.png'
    plt.savefig(output_path, dpi=600)
    logging.info(f"Saved {output_path}")

    # Close the plot
    plt.close(fig)

    return total_uncertainty_sum/background_int_excluded
#    return total_uncertainty_sum

def plot_signal_peak(channel, mll, hist_name, bins, mlljj_values, sig_mean, signal_stddev, signal_sample, output_dir=OUTPUT_DIR):
    fig, ax = plt.subplots()

    bins = np.arange(0, 8010, 10)
    hep.histplot(
        mlljj_values,
        bins=bins,
        histtype='step',
        color=['#5790fc'],
        label=[r"$(m_{W_R}, m_{N})=2000,1000 \mathrm{~GeV}$"],
        linewidth=2,
        ax=ax
    )

    # GUASSIAN FIT
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Compute bin centers
    initial_guess = [max(mlljj_values), np.mean(bin_centers), np.std(bin_centers)]  # Initial guess for fit parameters
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, mlljj_values, p0=initial_guess)
        amplitude, mean, stddev = popt

        # PLOT GAUSSIAN FIT
        x_fit = np.linspace(0, 4000, 1000)  # Generate x values for the fitted curve
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='red', linestyle='-', lw=2)

        # PLOT GAUSSIAN MEAN AND STDDEV
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.5)
        gaussian_at_mean = gaussian(mean, *popt)
        ymax_scaled = gaussian_at_mean / ax.get_ylim()[1]  # Scale the height of the line to fit within the y-axis limits
        ax.axvline(mean, color='red', linestyle='--', ymax=ymax_scaled, lw=2, label=f'Fit $\mu$ = {mean:.2f} GeV')

        x_stddev_fill = np.linspace(mean - (2*stddev), mean + (2*stddev), 100)
        y_stddev_fill = gaussian(x_stddev_fill, *popt)
        ax.fill_between(x_stddev_fill, y_stddev_fill, color='red', alpha=0.3, label=f'Fit $2\sigma$ = {abs(2*stddev):.2f} GeV')

    except RuntimeError as e:
        logging.error(f"Failed to fit Gaussian: {e}")

    # PLOT SIGNAL MEAN AND STDDEV
    ax.axvline(x=sig_mean, ymin=0, ymax=0.4, color='#5790fc', linestyle='--', lw=2, label=f'Sig. $\mu$ = {sig_mean:.2f} GeV')
    x_sig_stddev_fill = np.linspace(sig_mean - signal_stddev, sig_mean + signal_stddev, 100)
    y_sig_stddev_fill = np.interp(x_sig_stddev_fill, bin_centers, mlljj_values)
    ax.fill_between(x_sig_stddev_fill, 0, y_sig_stddev_fill, color='#5790fc', alpha=0.3, label=f'Sig. $2\sigma$ = {abs(2*signal_stddev):.2f} GeV')

    # ADD THE LABEL (Lower, Upper)
    lower_bound = mean - abs(2 * stddev)
    upper_bound = mean + abs(2 * stddev)
    ax.text(0.545, 0.74, f'(Low, High) = ({lower_bound:.2f}, {upper_bound:.2f})',
            transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')

    setup_plot(ax, "$m_{lljj} \mathrm{~[GeV]}$", r"Events / 10 GeV", xlim=(0, 4000))

    # Add CMS label and text box
    hep.cms.label(loc=2, ax=ax)
    flavor = r"$\mu\mu$" if channel == "mumujj" else r"$ee$"
    ax.text(0.05, 0.83, flavor, transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')

    # Create custom legend with specific font size
    legend_fontsize = 16  # Set desired legend font size here
    legend = ax.legend(fontsize=legend_fontsize, loc='upper right')  # Apply font size to the legend

    output_dir = f'{output_dir}/plots/mll{mll}'
    # Ensure the directory exists before saving the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = f'{output_dir}/{signal_sample}_mlljj_mll{mll}_{channel}.png'
    plt.savefig(output_path, dpi=600)
    logging.info(f"Saved {output_path}")

    # Close the plot
    plt.close(fig)

def plot_efficiency_vs_thresholds(channel, mll, hist_name, thresholds, efficiencies, efficiency_errors, output_dir="plotting/mll_optimization_study"):
    fig, ax = plt.subplots()

    # Plot efficiency using histplot
    hep.histplot(
        efficiencies,
        bins=thresholds,  # Use x_edges excluding the last edge to match y_vals length
        histtype='errorbar',
        yerr=efficiency_errors,
        xerr=True,
        color=['#5790fc'],
        label=[r"$(m_{W_R}, m_{N})=2000,1000 \mathrm{~GeV}$"],
        linewidth=2,
        ax=ax
    )

#    setup_plot(ax, "$m_{ll} \mathrm{~[GeV]}$", r"$\epsilon/(3/2 + \sqrt{B})$", xlim=(150, 1000), ylim=(0, 0.1))

    ax.set_xlabel("$m_{ll} \mathrm{~[GeV]}$")
#    ax.set_ylabel(r"$\epsilon/(3/2 + \sqrt{B})$")
    ax.set_ylabel(r"$\epsilon/(3/2 + \sqrt{B} + \sigma_{B})$")
    ax.set_ylim(0, 0.05)
    ax.legend(loc='best')

    # Add CMS label and text box
    hep.cms.label(loc=2, ax=ax)
    flavor = r"$\mu\mu$" if channel == "mumujj" else r"$ee$"
    ax.text(0.05, 0.83, flavor, transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')

    # Save the plot
    output_path = f'{output_dir}/plots/eff_over_rootB_{channel}.png'
    plt.savefig(output_path, dpi=600)
    logging.info(f"Saved {output_path}")

    plt.close(fig)

if __name__ == "__main__":
    try:
        # Ensure output directory exists
        ensure_directory_exists(OUTPUT_DIR)

        # Open the ROOT files for background and signal histograms
        bkg_root_file = open_root_file(BKG_FILE_PATH)
        signal_root_file = open_root_file(SIG_FILE_PATH)

        # Get and organize histograms from the ROOT files
        bkg_histograms = get_histograms(bkg_root_file)
        signal_histograms = get_histograms(signal_root_file)

        organized_bkg_histograms = organize_histograms(bkg_histograms)
        organized_signal_histograms = organize_histograms(signal_histograms)

        # Process histograms based on the defined criteria
        process_histograms(organized_bkg_histograms, organized_signal_histograms)

        logging.info("Finished processing histograms.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
