# SuMo-net

This is the code repository for Sumo-net proposed in the paper:

"Survival Regression with Proper Scoring Rules and Monotonic Neural Networks".

For a quick background/presentation, we refer to our talk at AISTATS 2022: 



To quickly run a model on a dataset, install dependencies, fill out the parameters in debug_run.py and simply run the file for a test run.

To install as a package, do:
```
git clone https://github.com/MrHuff/Sumo-Net.git
python setup.py bdist_wheel
pip install /dist/SuMo_net-0.1-py3-none-any.whl
```

To cite:
```

@InProceedings{pmlr-v151-rindt22a,
  title = 	 { Survival regression with proper scoring rules and monotonic neural networks },
  author =       {Rindt, David and Hu, Robert and Steinsaltz, David and Sejdinovic, Dino},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {1190--1205},
  year = 	 {2022},
  editor = 	 {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 Mar},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v151/rindt22a/rindt22a.pdf},
  url = 	 {https://proceedings.mlr.press/v151/rindt22a.html},
  abstract = 	 { We consider frequently used scoring rules for right-censored survival regression models such as time-dependent concordance, survival-CRPS, integrated Brier score and integrated binomial log-likelihood, and prove that neither of them is a proper scoring rule. This means that the true survival distribution may be scored worse than incorrect distributions, leading to inaccurate estimation. We prove, in contrast to these scores, that the right-censored log-likelihood is a proper scoring rule, i.e. the highest expected score is achieved by the true distribution. Despite this, modern feed-forward neural-network-based survival regression models are unable to train and validate directly on right-censored log-likelihood, due to its intractability, and resort to the aforementioned alternatives, i.e. non-proper scoring rules. We therefore propose a simple novel survival regression method capable of directly optimizing log-likelihood using a monotonic restriction on the time-dependent weights, coined SurvivalMonotonic-net (SuMo-net). SuMo-net achieves state-of-the-art log-likelihood scores across several datasets with 20–100x computational speedup on inference over existing state-of-the-art neural methods and is readily applicable to datasets with several million observations. }
}

```
This library is occasionally (as in almost never) maintained by Rob (MrHuff). Please raise an issue if there are any bugs, and it'll hopefully be taken care off.

If it's really urgent (i.e. your experiments aren't working or you are deploying something in healthcare), you can reach me at:

robert (dot) hu (at) stats (dot) ox (dot) ac (dot) uk

