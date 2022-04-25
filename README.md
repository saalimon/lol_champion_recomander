# League of legend recomendation model

League of legend recomendation model is a data science project including winrate prediction that we engieering the feature from <br>Kinkade, N. and Yul, K., 2015. DOTA 2 Win Prediction. [online](Jmcauley.ucsd.edu). Available at: <http://jmcauley.ucsd.edu/cse258/projects/fa15/018.pdf> and make the recomendation model based on unsupervised learning model (e.g. K-nearest neightbor) and association rule based (Apriori algorithm)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required library.

```bash
pip install requirements.txt
```

## Usage

```bash
flask run
```

## Experiment

For our experimnet the accuracy for winrate prediction model, we improve the accuracy from <strong>0.55 to 0.67</strong> (using <strong>decision tree</strong>) and test accuracy from <strong>0.62 to 0.72</strong> (using <strong>gradient boosting</strong>) by using <strong>champion synegy/countering</strong> on blue side and red side.<br>

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
