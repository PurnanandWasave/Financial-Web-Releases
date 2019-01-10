import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_och1\ as quotes_yahoo
from hmmlearn.hmm import GaussianHMM

#Load historical data
start = datetime.date(1970, 9, 4)
end = datetime(2018, 12, 31)
stock_quotes = quotes_yahoo('INTC', start, end)

#Daily quotes
closing_quotes = np.array([quote[2] for quote in stock_quotes])
volumes = np.array([quote[5] for quote in stock_quotes])[1:]

diff_percentages = 100.0 * np.diff(closing_quotes) / closing_quotes[ :-1]
dates = np.array([quote[0] for quote in stock_quotes], dtype=np.int)[1:]

training_data = np.column_stack([diff_percentages, volumes])

#HMM model
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000)
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  hmm.fit(training_data)

#Inputing samples
num_samples = 500
samples, _ = hmm.sample(num_samples)

#Plot graph
plt.figure()
plt.title("Difference in percentages")
plt.plot(np.arrange(num_samples), samples[:, 0], c='black')

plt.figure()
plt.title('Volumes of the stocks')
plt.plot(np.arrange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)

plt.show()


##Stock Market analysis system is ready##








