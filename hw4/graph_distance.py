import numpy as np
import csv
import matplotlib.pyplot as plt
from dataGenerator import data_generator
from scipy.optimize import curve_fit
import warnings
import math

def func(x, a, b, c):
    return  - a*np.exp(-b*x) + c
def func2(y, a, b, c):
    res = -1./b*np.log((c-y)/a)
    return 60 if math.isnan(res) else res

def poly_fit(x, y, degree, fit="RANSAC"):
  # check if we can use RANSAC
  if fit == "RANSAC":
    try:
      import sklearn.linear_model as sklin
      import sklearn.preprocessing as skpre
    except ImportError:
      warnings.warn(
        "fitting mode 'RANSAC' requires the package sklearn, using"
        + "'poly' instead",
        RuntimeWarning)
      fittig_mode = "poly"

  if fit == "poly":
    return np.polyfit(x, y, degree)
  elif fit == "RANSAC":
    model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
    xdat = np.array(x)
    if len(xdat.shape) == 1:
      # interpret 1d-array as list of len(x) samples instead of
      # one sample of length len(x)
      xdat = xdat.reshape(-1, 1)
    polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
    # evil hack to circumvent model.fit failing if no inliers are found
    # in one iteration: repeat model.fit up to 10 times
    success = False
    for _ in range(10):
      try:
        model.fit(polydat, y)
        success = True
        break
      except ValueError:
        pass
    # fall back to polyfit if all 10 iterations failed
    if not success:
      warnings.warn(
        "RANSAC did not reach consensus, "
        + "using numpy's polyfit",
        RuntimeWarning)
      coef = np.polyfit(x, y, degree)
    else:
      coef = model.estimator_.coef_[::-1]
    return coef
  else:
    raise ValueError("invalid fitting mode ({})".format(fit))


def plot_reg(xvals, yvals, poly, x_label="x", y_label="y", data_label="data",
             reg_label="regression line", fname=None):
  """
  Helper function to plot trend lines for line-fitting approaches. This
  function will show a plot through `plt.show()` and close it after the window
  has been closed by the user.
  Args:
    xvals (list/array of float):
      list of x-values
    yvals (list/array of float):
      list of y-values
    poly (list/array of float):
      polynomial parameters as accepted by `np.polyval`
  Kwargs:
    x_label (str):
      label of the x-axis
    y_label (str):
      label of the y-axis
    data_label (str):
      label of the data
    reg_label(str):
      label of the regression line
    fname (str):
      file name (if not None, the plot will be saved to disc instead of
      showing it though `plt.show()`)
  """
  # local import to avoid dependency for non-debug use
  plt.plot(xvals, yvals, "bo", label=data_label)
  if not (poly is None):
    plt.plot(xvals, np.polyval(poly, xvals), "r-", label=reg_label)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc="best")
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()

def logarithmic_r(min_n, max_n, factor):
  """
  Creates a list of values by successively multiplying a minimum value min_n by
  a factor > 1 until a maximum value max_n is reached.
  Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)
  Returns:
    list of floats:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
  """
  assert max_n > min_n
  assert factor > 1
  max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
  return [min_n * (factor ** i) for i in range(max_i + 1)]

def dist(X, y):
    return [np.linalg.norm(x-y) for x in X]

def GP(data, rvals=None, plot_file=None):

  n = len(data)
  #orbit = np.array([data[i:i + emb_dim] for i in range(n - emb_dim + 1)])
  orbit = data
  dists = np.array([dist(orbit, orbit[i]) for i in range(len(orbit))])
  mn = min(dists.flatten())+1
  mx = max(dists.flatten())
#  print(mn, mx)
  if rvals is None:
    sd = np.std(data.flatten())
    rvals = logarithmic_r( 6*sd, 13*sd, 1.03)

  csums = []
  for r in rvals:
    s = 1.0 / (n * (n - 1)) * np.sum(dists < r)
    csums.append(s)
  csums = np.array(csums)
  # filter zeros from csums
  nonzero = np.where(csums != 0)
  rvals = np.array(rvals)[nonzero]
  csums = csums[nonzero]
  if len(csums) == 0:
    # all sums are zero => we cannot fit a line
    poly = [np.nan, np.nan]
  else:
    poly = poly_fit(np.log(rvals), np.log(csums), 1)

  debug_plot = False
  if debug_plot:
    plot_reg(np.log(rvals), np.log(csums), poly, "log(r)", "log(C(r))", fname=plot_file)
  return poly[0]


outputFile="res.csv"
def writeResult(coe):
    X = np.arange(100)
    Y = [func(x, coe[0], coe[1], coe[2]) for x in X]
    plt.plot(X, Y)
    plt.show()

    all_data = np.load('q3/data.npz')
    with open(outputFile, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['setId', 'LogDim'])
        for i in range(200):
            gp = GP(all_data[str(i)][:5000])
            print(gp)
            res = func2(gp, coe[0], coe[1], coe[2])
            print(i,res)
            csv_writer.writerow([i]+[np.log(res)])

def main():
    generator = data_generator()
    '''
    X = np.arange(1, 61)
    Y = []
    for i in X:
        res = GP(generator.generate(i,5000))
        print(res)
        Y.append(res)
    plt.plot(X, Y)
    plt.savefig("haha")
    '''
    X = np.arange(1, 61, 4)

    Y = [0.790993770053,
         3.52745718491,
         5.09861619009,
         6.81841850028,
         7.55939510887,
         7.97493702122,
         8.91988878073,
         8.56210922693,
         8.15322864393,
         8.73514918244,
         8.50389504511,
         9.63617578701,
         8.83023870185,
         9.0094963608,
         9.48107348887 ]


    coe = curve_fit(func, X, Y)[0]
    for i in range(61):
        gp = GP( generator.generate(i, 1000))
        print( func2(gp, coe[0], coe[1], coe[2]) )
#    writeResult(coe[0])
#    Y = [GP(generator.generate(i, 1000)) for i in X]

main()


